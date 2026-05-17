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
exports.Breadcrumb = void 0;
var tslib_1 = require("tslib");
var classnames_1 = tslib_1.__importDefault(require("classnames"));
var React = tslib_1.__importStar(require("react"));
var common_1 = require("../../common");
var icon_1 = require("../icon/icon");
/**
 * Breadcrumb component.
 *
 * @see https://blueprintjs.com/docs/#core/components/breadcrumbs
 */
var Breadcrumb = function (props) {
    var _a;
    var classes = (0, classnames_1.default)(common_1.Classes.BREADCRUMB, (_a = {},
        _a[common_1.Classes.BREADCRUMB_CURRENT] = props.current,
        _a[common_1.Classes.DISABLED] = props.disabled,
        _a), props.className);
    var icon = props.icon != null ? React.createElement(icon_1.Icon, { title: props.iconTitle, icon: props.icon }) : undefined;
    if (props.href == null && props.onClick == null) {
        return (React.createElement("span", { className: classes },
            icon,
            props.text,
            props.children));
    }
    return (React.createElement("a", { className: classes, href: props.href, onClick: props.disabled ? undefined : props.onClick, onFocus: props.disabled ? undefined : props.onFocus, tabIndex: props.disabled ? undefined : 0, target: props.target },
        icon,
        props.text,
        props.children));
};
exports.Breadcrumb = Breadcrumb;
//# sourceMappingURL=breadcrumb.js.map