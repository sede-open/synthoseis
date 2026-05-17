/*
 * Copyright 2020 Palantir Technologies, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { __assign, __rest } from "tslib";
import * as React from "react";
import { AnchorButton } from "../button/buttons";
import { Tooltip } from "../tooltip/tooltip";
export function DialogStepButton(_a) {
    var tooltipContent = _a.tooltipContent, props = __rest(_a, ["tooltipContent"]);
    var button = React.createElement(AnchorButton, __assign({}, props));
    if (tooltipContent !== undefined) {
        return React.createElement(Tooltip, { content: tooltipContent }, button);
    }
    else {
        return button;
    }
}
//# sourceMappingURL=dialogStepButton.js.map