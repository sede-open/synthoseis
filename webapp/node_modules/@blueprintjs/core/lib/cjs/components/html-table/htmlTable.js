"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.HTMLTable = void 0;
var tslib_1 = require("tslib");
var classnames_1 = tslib_1.__importDefault(require("classnames"));
var React = tslib_1.__importStar(require("react"));
var common_1 = require("../../common");
// this component is simple enough that tests would be purely tautological.
/* istanbul ignore next */
/**
 * HTML table component.
 *
 * @see https://blueprintjs.com/docs/#core/components/html-table
 */
exports.HTMLTable = React.forwardRef(function (props, ref) {
    var _a;
    var bordered = props.bordered, className = props.className, compact = props.compact, interactive = props.interactive, striped = props.striped, htmlProps = tslib_1.__rest(props, ["bordered", "className", "compact", "interactive", "striped"]);
    var classes = (0, classnames_1.default)(common_1.Classes.HTML_TABLE, (_a = {},
        _a[common_1.Classes.COMPACT] = compact,
        _a[common_1.Classes.HTML_TABLE_BORDERED] = bordered,
        _a[common_1.Classes.HTML_TABLE_STRIPED] = striped,
        _a[common_1.Classes.INTERACTIVE] = interactive,
        _a), className);
    // eslint-disable-next-line @blueprintjs/html-components
    return React.createElement("table", tslib_1.__assign({}, htmlProps, { ref: ref, className: classes }));
});
exports.HTMLTable.displayName = "".concat(common_1.DISPLAYNAME_PREFIX, ".HTMLTable");
//# sourceMappingURL=htmlTable.js.map