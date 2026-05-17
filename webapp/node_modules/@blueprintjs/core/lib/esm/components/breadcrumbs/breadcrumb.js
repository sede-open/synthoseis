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
import classNames from "classnames";
import * as React from "react";
import { Classes } from "../../common";
import { Icon } from "../icon/icon";
/**
 * Breadcrumb component.
 *
 * @see https://blueprintjs.com/docs/#core/components/breadcrumbs
 */
export var Breadcrumb = function (props) {
    var _a;
    var classes = classNames(Classes.BREADCRUMB, (_a = {},
        _a[Classes.BREADCRUMB_CURRENT] = props.current,
        _a[Classes.DISABLED] = props.disabled,
        _a), props.className);
    var icon = props.icon != null ? React.createElement(Icon, { title: props.iconTitle, icon: props.icon }) : undefined;
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
//# sourceMappingURL=breadcrumb.js.map