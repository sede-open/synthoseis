/* !
 * (c) Copyright 2022 Palantir Technologies Inc. All rights reserved.
 */
import { __assign, __rest } from "tslib";
import * as React from "react";
import { Classes, DISPLAYNAME_PREFIX } from "../../common";
export var ResizableInput = React.forwardRef(function ResizableInput(props, ref) {
    var _a = React.useState(""), content = _a[0], setContent = _a[1];
    var _b = React.useState(0), width = _b[0], setWidth = _b[1];
    var span = React.useRef(null);
    React.useEffect(function () {
        if (span.current != null) {
            setWidth(span.current.offsetWidth);
        }
    }, [content]);
    var onChange = props.onChange, style = props.style, otherProps = __rest(props, ["onChange", "style"]);
    var handleInputChange = function (evt) {
        var _a, _b;
        onChange === null || onChange === void 0 ? void 0 : onChange(evt);
        setContent((_b = (_a = evt === null || evt === void 0 ? void 0 : evt.target) === null || _a === void 0 ? void 0 : _a.value) !== null && _b !== void 0 ? _b : "");
    };
    return (React.createElement(React.Fragment, null,
        React.createElement("span", { ref: span, className: Classes.RESIZABLE_INPUT_SPAN, "aria-hidden": true }, content.replace(/ /g, "\u00a0")),
        React.createElement("input", __assign({}, otherProps, { type: "text", style: __assign(__assign({}, style), { width: width }), onChange: handleInputChange, ref: ref }))));
});
ResizableInput.displayName = "".concat(DISPLAYNAME_PREFIX, ".ResizableInput");
//# sourceMappingURL=resizableInput.js.map