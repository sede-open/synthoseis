/*
 * Copyright 2021 Palantir Technologies, Inc. All rights reserved.
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
import classNames from "classnames";
import * as React from "react";
import { Classes } from "../../common";
import { Dialog } from "../dialog/dialog";
import { DialogBody } from "../dialog/dialogBody";
import { Hotkey } from "./hotkey";
import { Hotkeys } from "./hotkeys";
export var HotkeysDialog2 = function (_a) {
    var _b = _a.globalGroupName, globalGroupName = _b === void 0 ? "Global" : _b, hotkeys = _a.hotkeys, props = __rest(_a, ["globalGroupName", "hotkeys"]);
    return (React.createElement(Dialog, __assign({}, props, { className: classNames(Classes.HOTKEY_DIALOG, props.className) }),
        React.createElement(DialogBody, null,
            React.createElement(Hotkeys, null, hotkeys.map(function (hotkey, index) { return (React.createElement(Hotkey, __assign({ key: index }, hotkey, { group: hotkey.global === true && hotkey.group == null ? globalGroupName : hotkey.group }))); })))));
};
//# sourceMappingURL=hotkeysDialog2.js.map