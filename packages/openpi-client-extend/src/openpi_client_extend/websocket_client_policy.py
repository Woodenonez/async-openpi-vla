from typing import Dict

from openpi_client.websocket_client_policy import WebsocketClientPolicy


class WebsocketClientPolicyExtend(WebsocketClientPolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def infer_batch(self, obs: Dict, nbatch: None | int = 1) -> Dict:  # noqa: UP006
        if isinstance(nbatch, int):
            obs = dict(obs)  # shallow copy
            obs["__nbatch__"] = nbatch
            
        return self.infer(obs)

