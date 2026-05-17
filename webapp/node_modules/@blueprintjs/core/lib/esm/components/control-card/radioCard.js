/*
 * Copyright 2023 Palantir Technologies, Inc. All rights reserved.
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
import { __assign } from "tslib";
import classNames from "classnames";
import * as React from "react";
import { Classes } from "../../common";
import { DISPLAYNAME_PREFIX } from "../../common/props";
import { ControlCard } from "./controlCard";
/**
 * Radio Card component.
 *
 * @see https://blueprintjs.com/docs/#core/components/control-card.radio-card
 */
export var RadioCard = React.forwardRef(function (props, ref) {
    var className = classNames(props.className, Classes.RADIO_CONTROL_CARD);
    return React.createElement(ControlCard, __assign({}, props, { className: className, controlKind: "radio", ref: ref }));
});
RadioCard.displayName = "".concat(DISPLAYNAME_PREFIX, ".RadioCard");
//# sourceMappingURL=radioCard.js.map