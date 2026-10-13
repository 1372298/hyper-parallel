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
"""Exercise the GSM8K reward function used by the shipped agent."""

import pytest

from examples.gsm8k.agent import compute_gsm8k_reward


@pytest.mark.parametrize(
    ("response", "target", "expected"),
    [
        ("work\n#### 42", "42", 1.0),
        ("first #### 1\nfinal #### -12", "-12", 1.0),
        ("#### 1,024", "$1,024", 1.0),
        (" 3.14 ", "3.14", 1.0),
        ("#### 2", "3", 0.0),
        ("#### 2" + "x" * 300, "2", 0.0),
        ("work\n#### $18", "18", 1.0),
        ("#### 18.00", "18", 1.0),
        ("#### $1,024.50", "1024.5", 1.0),
        ("####\n+18", "18", 1.0),
        ("#### -.5", "-0.50", 1.0),
        ("#### -0.0", "0", 1.0),
        ("#### 123456789012345678901234567890.00", "123456789012345678901234567890", 1.0),
        ("#### 9007199254740993", "9007199254740992", 0.0),
        ("#### 18/2", "18", 0.0),
        ("#### 18.2.3", "18.2", 0.0),
        ("#### 1,20", "120", 0.0),
        ("#### nan", "nan", 0.0),
        ("#### 18\nfinal #### unknown", "18", 0.0),
    ],
)
def test_shipped_gsm8k_reward(response: str, target: str, expected: float) -> None:
    """Score final or direct numeric answers using the actual training reward."""
    assert compute_gsm8k_reward(response, target) == expected
