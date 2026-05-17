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
import { __assign, __extends, __rest } from "tslib";
import classNames from "classnames";
import * as React from "react";
import { AbstractPureComponent, Classes, DISPLAYNAME_PREFIX } from "../../common";
import { KeyComboTag } from "./keyComboTag";
/**
 * Hotkey component used to display a hotkey in the HotkeysDialog.
 * Should not be used by consumers directly.
 */
var Hotkey = /** @class */ (function (_super) {
    __extends(Hotkey, _super);
    function Hotkey() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Hotkey.prototype.render = function () {
        var _a = this.props, label = _a.label, className = _a.className, spreadableProps = __rest(_a, ["label", "className"]);
        var rootClasses = classNames(Classes.HOTKEY, className);
        return (React.createElement("div", { className: rootClasses },
            React.createElement("div", { className: Classes.HOTKEY_LABEL }, label),
            React.createElement(KeyComboTag, __assign({}, spreadableProps))));
    };
    Hotkey.prototype.validateProps = function (props) {
        if (props.global !== true && props.group == null) {
            console.error("non-global Hotkeys must define a group");
        }
    };
    Hotkey.displayName = "".concat(DISPLAYNAME_PREFIX, ".Hotkey");
    Hotkey.defaultProps = {
        allowInInput: false,
        disabled: false,
        global: false,
        preventDefault: false,
        stopPropagation: false,
    };
    return Hotkey;
}(AbstractPureComponent));
export { Hotkey };
//# sourceMappingURL=hotkey.js.map