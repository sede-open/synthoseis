/*
 * Copyright 2018 Palantir Technologies, Inc. All rights reserved.
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
import { Classes, DISPLAYNAME_PREFIX } from "../../common";
// this component is simple enough that tests would be purely tautological.
/* istanbul ignore next */
/**
 * HTML table component.
 *
 * @see https://blueprintjs.com/docs/#core/components/html-table
 */
export var HTMLTable = React.forwardRef(function (props, ref) {
    var _a;
    var bordered = props.bordered, className = props.className, compact = props.compact, interactive = props.interactive, striped = props.striped, htmlProps = __rest(props, ["bordered", "className", "compact", "interactive", "striped"]);
    var classes = classNames(Classes.HTML_TABLE, (_a = {},
        _a[Classes.COMPACT] = compact,
        _a[Classes.HTML_TABLE_BORDERED] = bordered,
        _a[Classes.HTML_TABLE_STRIPED] = striped,
        _a[Classes.INTERACTIVE] = interactive,
        _a), className);
    // eslint-disable-next-line @blueprintjs/html-components
    return React.createElement("table", __assign({}, htmlProps, { ref: ref, className: classes }));
});
HTMLTable.displayName = "".concat(DISPLAYNAME_PREFIX, ".HTMLTable");
//# sourceMappingURL=htmlTable.js.map