# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindSpore object-list communication helpers."""
import io
import pickle
from typing import Any, Optional


def send_object_list(obj: list[Any], dst: int = 0, group: Optional[str] = None) -> None:
    """
    Send the input Python object to dst rank.

    Args:
        obj (Any): The input tensor to be send.
        dst (int, optional): Specifies the global rank that send the Python object to.
            Default: ``0``.
        group (str, optional): Communication group. Default: ``None``.
    """
    from mindspore import Tensor  # pylint: disable=C0415
    from mindspore.common import dtype as mstype  # pylint: disable=C0415
    from mindspore.communication import GlobalComm  # pylint: disable=C0415
    from mindspore.mint.distributed.distributed import _object_to_tensor, send  # pylint: disable=C0415
    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP
    if not isinstance(group, str):
        raise TypeError(f"For 'send_object', the argument 'group' must be type of string, \
                          but got 'group' type : {type(group)}.")
    if not isinstance(dst, int):
        raise TypeError("For send_object, the dst must be int.")
    obj_tensor, tensor_size = _object_to_tensor(obj)
    obj_size = Tensor([tensor_size], dtype=mstype.int32)
    send(obj_size, dst, group)
    send(obj_tensor, dst, group)



def recv_object_list(recv_obj: list[Any], src: int = 0, group: Optional[str] = None) -> None:
    """
    receive Python object from src rank.

    Args:
        recv_obj (list): list to recv python objects.
        src (int, optional): Specifies the global rank that receive the Python object.
            Default: ``0`` .
        group (str, optional): Communication group. Default: ``None``.
    """
    from mindspore import mint  # pylint: disable=C0415
    from mindspore.common import dtype as mstype  # pylint: disable=C0415
    from mindspore.communication import GlobalComm  # pylint: disable=C0415
    from mindspore.mint.distributed.distributed import recv  # pylint: disable=C0415
    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP
    if not isinstance(group, str):
        raise TypeError(f"For 'recv_object', the argument 'group' must be type of string, \
                          but got 'group' type : {type(group)}.")
    if not isinstance(src, int):
        raise TypeError("For recv_object, the src must be int.")
    obj_size = mint.zeros((1,), dtype=mstype.int32)
    recv(obj_size, src, group)
    # MindSpore PyNative ``recv`` only does a comm-stream wait; bridge to host
    # so the subsequent ``.item()`` reads the freshly-received value instead
    # of the original buffer.
    size_val = int(obj_size.item())
    obj_tensor = mint.zeros((size_val,), dtype=mstype.int8)
    recv(obj_tensor, src, group)
    buf = obj_tensor.asnumpy().tobytes()[:size_val]
    recv_obj.clear()
    recv_obj.append(pickle.Unpickler(io.BytesIO(buf)).load()[0])
