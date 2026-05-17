"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.DialogStepButton = void 0;
var tslib_1 = require("tslib");
var React = tslib_1.__importStar(require("react"));
var buttons_1 = require("../button/buttons");
var tooltip_1 = require("../tooltip/tooltip");
function DialogStepButton(_a) {
    var tooltipContent = _a.tooltipContent, props = tslib_1.__rest(_a, ["tooltipContent"]);
    var button = React.createElement(buttons_1.AnchorButton, tslib_1.__assign({}, props));
    if (tooltipContent !== undefined) {
        return React.createElement(tooltip_1.Tooltip, { content: tooltipContent }, button);
    }
    else {
        return button;
    }
}
exports.DialogStepButton = DialogStepButton;
//# sourceMappingURL=dialogStepButton.js.map