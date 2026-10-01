"""Head phan loai: vector dac trung phang -> logit n_classes.

mode='classical': MLP.
mode='quantum': RESIDUAL FUSION (theo QEP) -> on dinh + quantum bo sung dac trung:
    x -> [ VQC(12 obs)  ||  bypass classical ] -> MLP.
Nhanh bypass giu duong classical luon co (giam phuong sai); VQC them dac trung phi tuyen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pennylane as qml
    _HAS_QML = True
except Exception:
    _HAS_QML = False


def build_qnode(n_qubits=6, n_layers=2, encoding="amplitude"):
    dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(inputs, weights):
        if encoding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[l, q, 0], wires=q)
                qml.RZ(weights[l, q, 1], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        z = [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        zz = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % n_qubits)) for q in range(n_qubits)]
        return z + zz

    return qml.QNode(circuit, dev, interface="torch", diff_method="backprop")


class QuantumClassifierHead(nn.Module):
    def __init__(self, in_features, n_qubits=6, n_layers=2, encoding="amplitude",
                 hidden=128, n_classes=4, mode="quantum", bypass_dim=64):
        super().__init__()
        self.mode = mode
        self.encoding = encoding
        self.n_qubits = n_qubits

        if mode == "classical":
            self.compress = nn.Sequential(nn.Linear(in_features, 128), nn.GELU())
            self.head = nn.Sequential(nn.Linear(128, hidden), nn.GELU(), nn.Linear(hidden, n_classes))
            return

        if not _HAS_QML:
            raise ImportError("Chua cai PennyLane. Chay: pip install pennylane")
        feat_in = (2 ** n_qubits) if encoding == "amplitude" else n_qubits
        self.to_q = nn.Linear(in_features, feat_in)
        qnode = build_qnode(n_qubits, n_layers, encoding)
        self.vqc = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 2)})
        self.bypass = nn.Sequential(nn.Linear(in_features, bypass_dim), nn.GELU())   # duong classical song song
        self.head = nn.Sequential(nn.Linear(2 * n_qubits + bypass_dim, hidden),
                                  nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, x):                   # x: (B, in_features)
        if self.mode == "classical":
            return self.head(self.compress(x))
        q_in = self.to_q(x)
        if self.encoding == "amplitude":
            q_in = F.normalize(q_in, dim=-1) + 1e-12
        q = self.vqc(q_in)                  # (B, 12) dac trung luong tu
        b = self.bypass(x)                  # (B, bypass_dim) classical
        return self.head(torch.cat([q, b], dim=1))
