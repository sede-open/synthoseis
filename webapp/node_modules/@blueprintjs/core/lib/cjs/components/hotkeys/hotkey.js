"use strict";
/*
 * Copyright 2016 Palantir Technologies, Inc. All rights reserved.
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
exports.Hotkey = void 0;
var tslib_1 = require("tslib");
var classnames_1 = tslib_1.__importDefault(require("classnames"));
var React = tslib_1.__importStar(require("react"));
var common_1 = require("../../common");
var keyComboTag_1 = require("./keyComboTag");
/**
 * Hotkey component used to display a hotkey in the HotkeysDialog.
 * Should not be used by consumers directly.
 */
var Hotkey = /** @class */ (function (_super) {
    tslib_1.__extends(Hotkey, _super);
    function Hotkey() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Hotkey.prototype.render = function () {
        var _a = this.props, label = _a.label, className = _a.className, spreadableProps = tslib_1.__rest(_a, ["label", "className"]);
        var rootClasses = (0, classnames_1.default)(common_1.Classes.HOTKEY, className);
        return (React.createElement("div", { className: rootClasses },
            React.createElement("div", { className: common_1.Classes.HOTKEY_LABEL }, label),
            React.createElement(keyComboTag_1.KeyComboTag, tslib_1.__assign({}, spreadableProps))));
    };
    Hotkey.prototype.validateProps = function (props) {
        if (props.global !== true && props.group == null) {
            console.error("non-global Hotkeys must define a group");
        }
    };
    Hotkey.displayName = "".concat(common_1.DISPLAYNAME_PREFIX, ".Hotkey");
    Hotkey.defaultProps = {
        allowInInput: false,
        disabled: false,
        global: false,
        preventDefault: false,
        stopPropagation: false,
    };
    return Hotkey;
}(common_1.AbstractPureComponent));
exports.Hotkey = Hotkey;
//# sourceMappingURL=hotkey.js.map