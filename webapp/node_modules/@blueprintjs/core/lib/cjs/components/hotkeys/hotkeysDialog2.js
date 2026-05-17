"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.HotkeysDialog2 = void 0;
var tslib_1 = require("tslib");
var classnames_1 = tslib_1.__importDefault(require("classnames"));
var React = tslib_1.__importStar(require("react"));
var common_1 = require("../../common");
var dialog_1 = require("../dialog/dialog");
var dialogBody_1 = require("../dialog/dialogBody");
var hotkey_1 = require("./hotkey");
var hotkeys_1 = require("./hotkeys");
var HotkeysDialog2 = function (_a) {
    var _b = _a.globalGroupName, globalGroupName = _b === void 0 ? "Global" : _b, hotkeys = _a.hotkeys, props = tslib_1.__rest(_a, ["globalGroupName", "hotkeys"]);
    return (React.createElement(dialog_1.Dialog, tslib_1.__assign({}, props, { className: (0, classnames_1.default)(common_1.Classes.HOTKEY_DIALOG, props.className) }),
        React.createElement(dialogBody_1.DialogBody, null,
            React.createElement(hotkeys_1.Hotkeys, null, hotkeys.map(function (hotkey, index) { return (React.createElement(hotkey_1.Hotkey, tslib_1.__assign({ key: index }, hotkey, { group: hotkey.global === true && hotkey.group == null ? globalGroupName : hotkey.group }))); })))));
};
exports.HotkeysDialog2 = HotkeysDialog2;
//# sourceMappingURL=hotkeysDialog2.js.map