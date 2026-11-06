"""
Loader for the pretrained sEH proxy shipped in
``envs/sEH_large/pretrained_proxy/``.

Reads ``info.pkl.gz`` to discover the architecture (``repr_type``,
``model_version``, ``nemb``, ``num_conv_steps``) the proxy was trained
with, instantiates the matching model, and copies the parameter arrays
from ``best_params.pkl.gz`` into it.
"""

import gzip
import os
import pickle
from typing import List, Optional

import numpy as np
import torch

from . import model_atom, model_block
from .mol_mdp_ext import MolMDPExtended


_DEFAULT_PROXY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pretrained_proxy"
)


def _load_eargs(proxy_dir: str):
    """Return the argparse.Namespace stored in info.pkl.gz['args']."""
    with gzip.open(os.path.join(proxy_dir, "info.pkl.gz"), "rb") as f:
        info = pickle.load(f)
    return info["args"]


def _make_proxy_model(eargs, mdp):
    repr_type = eargs.repr_type
    if repr_type == "atom_graph":
        model = model_atom.MolAC_GCN(
            nhid=eargs.nemb,
            nvec=0,
            num_out_per_stem=mdp.num_blocks,
            num_out_per_mol=1,
            num_conv_steps=eargs.num_conv_steps,
            version=eargs.model_version,
            dropout_rate=0.0,
        )
    elif repr_type == "block_graph":
        model = model_block.GraphAgent(
            nemb=eargs.nemb,
            nvec=0,
            out_per_stem=mdp.num_blocks,
            out_per_mol=1,
            num_conv_steps=eargs.num_conv_steps,
            mdp_cfg=mdp,
            version=eargs.model_version,
        )
    else:
        raise ValueError(
            f"Unsupported pretrained proxy repr_type {repr_type!r}; "
            "only 'atom_graph' and 'block_graph' are supported."
        )
    return model


class SEHGraphProxy:
    """Wraps the pretrained NN proxy from GFNOrg/gflownet for sEH.

    Use ``SEHGraphProxy.from_default()`` to load the proxy from the
    default ``envs/sEH_large/pretrained_proxy/`` directory.
    """

    def __init__(
        self,
        blocks_file: str,
        proxy_dir: str = _DEFAULT_PROXY_DIR,
        device: Optional[torch.device] = None,
        floatX: torch.dtype = torch.float32,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.floatX = floatX

        self.eargs = _load_eargs(proxy_dir)
        self.repr_type = self.eargs.repr_type

        self.mdp = MolMDPExtended(blocks_file)
        self.mdp.post_init(device, self.repr_type)
        self.mdp.floatX = floatX

        # Constructing the proxy initializes every layer, including the unused
        # bond head, before the pretrained tensors are copied in.  Isolate that
        # initialization from the experiment RNG so creating a different
        # number of environment/proxy instances cannot change the GFlowNet
        # policy initialization or action-sampling stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.model = _make_proxy_model(self.eargs, self.mdp)
        self._load_params(os.path.join(proxy_dir, "best_params.pkl.gz"))
        self.model.to(floatX)
        self.model.to(device)
        self.model.eval()

    # Names of the model parameters that correspond, in order, to the 22
    # tensors stored in the pretrained sEH proxy at
    # envs/sEH_large/pretrained_proxy/best_params.pkl.gz
    #
    # The proxy was trained against a circa-2022 version of GFNOrg/gflownet
    # that (a) used a slightly older PyG NNConv layout (``root_weight`` and
    # ``bias`` as standalone Parameters, with the root coming first), and
    # (b) did not yet include the ``bond2out`` head. We therefore map the
    # 22 saved tensors onto the 22 functionally-corresponding parameters
    # in the current model and leave ``bond2out`` randomly initialised
    # (it is unused by the proxy reward path, which calls the model with
    # ``do_bonds=False``).
    _ATOM_GRAPH_V2_PARAM_MAP = [
        ("mpnn.lin0.weight", None),
        ("mpnn.lin0.bias", None),
        # NNConv root weight and output bias
        ("mpnn.conv.lin.weight", None),
        ("mpnn.conv.bias", None),
        # NNConv edge MLP (Sequential[Linear, LeakyReLU, Linear])
        ("mpnn.conv.nn.0.weight", None),
        ("mpnn.conv.nn.0.bias", None),
        ("mpnn.conv.nn.2.weight", None),
        ("mpnn.conv.nn.2.bias", None),
        # GRU
        ("mpnn.gru.weight_ih_l0", None),
        ("mpnn.gru.weight_hh_l0", None),
        ("mpnn.gru.bias_ih_l0", None),
        ("mpnn.gru.bias_hh_l0", None),
        # Per-stem head
        ("mpnn.lin1.weight", None),
        ("mpnn.lin1.bias", None),
        ("mpnn.lin2.weight", None),
        ("mpnn.lin2.bias", None),
        # Set2Set (an LSTM with input dim 2*nemb, hidden dim nemb)
        ("mpnn.set2set.lstm.weight_ih_l0", None),
        ("mpnn.set2set.lstm.weight_hh_l0", None),
        ("mpnn.set2set.lstm.bias_ih_l0", None),
        ("mpnn.set2set.lstm.bias_hh_l0", None),
        # Per-molecule head (the actual proxy output)
        ("mpnn.lin3.weight", None),
        ("mpnn.lin3.bias", None),
    ]

    def _load_params(self, params_path: str):
        with gzip.open(params_path, "rb") as f:
            params: List[np.ndarray] = pickle.load(f)

        if self.repr_type == "atom_graph" and self.eargs.model_version == "v2":
            self._load_params_named(params, self._ATOM_GRAPH_V2_PARAM_MAP)
        else:
            self._load_params_positional(params)

    def _load_params_positional(self, params: List[np.ndarray]):
        own_params = list(self.model.parameters())
        if len(own_params) != len(params):
            raise RuntimeError(
                f"Pretrained proxy has {len(params)} parameter tensors but the "
                f"reconstructed {self.repr_type}/{self.eargs.model_version} model "
                f"has {len(own_params)}. Architecture mismatch — add an explicit "
                f"name-based mapping in proxy.py for this configuration."
            )
        for own, src in zip(own_params, params):
            src_t = torch.tensor(src, dtype=self.floatX)
            if tuple(own.shape) != tuple(src_t.shape):
                raise RuntimeError(
                    f"Shape mismatch loading proxy params: model expects "
                    f"{tuple(own.shape)}, file has {tuple(src_t.shape)}."
                )
            own.data = src_t

    def _load_params_named(self, params: List[np.ndarray], mapping):
        if len(params) != len(mapping):
            raise RuntimeError(
                f"Pretrained proxy has {len(params)} tensors but the "
                f"{self.repr_type}/{self.eargs.model_version} mapping has "
                f"{len(mapping)} entries."
            )
        named = dict(self.model.named_parameters())
        for src, (name, _) in zip(params, mapping):
            if name not in named:
                raise RuntimeError(
                    f"Parameter '{name}' not found in reconstructed model. "
                    f"PyG / torch may have changed the NNConv layout."
                )
            own = named[name]
            src_t = torch.tensor(src, dtype=self.floatX)
            if tuple(own.shape) != tuple(src_t.shape):
                raise RuntimeError(
                    f"Shape mismatch loading proxy param '{name}': model "
                    f"expects {tuple(own.shape)}, file has {tuple(src_t.shape)}."
                )
            own.data = src_t

    @torch.no_grad()
    def __call__(self, mol) -> float:
        """Return the proxy's per-molecule scalar output for one molecule."""
        batch = self.mdp.mols2batch([self.mdp.mol2repr(mol)])
        _, mol_out = self.model(batch, None, do_stems=False)
        return float(mol_out[0, 0].item())

    @torch.no_grad()
    def batch_score(self, mols) -> np.ndarray:
        """Score a list of molecules in a single forward pass."""
        if len(mols) == 0:
            return np.zeros(0, dtype=np.float32)
        batch = self.mdp.mols2batch([self.mdp.mol2repr(m) for m in mols])
        _, mol_out = self.model(batch, None, do_stems=False)
        return mol_out[:, 0].detach().cpu().numpy()
