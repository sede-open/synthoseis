/*
 * Copyright 2024 Palantir Technologies, Inc. All rights reserved.
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
import * as React from "react";
import { IconSize } from "../../iconTypes";
import { SVGIconContainer } from "../../svgIconContainer";
export const Modal = React.forwardRef((props, ref) => {
    const isLarge = props.size >= IconSize.LARGE;
    const pixelGridSize = isLarge ? IconSize.LARGE : IconSize.STANDARD;
    const translation = `${-1 * pixelGridSize / 0.05 / 2}`;
    const style = { transformOrigin: "center" };
    return (React.createElement(SVGIconContainer, { iconName: "modal", ref: ref, ...props },
        React.createElement("path", { d: isLarge ? "M379.999998 380C391.0456940000001 380 399.999998 371.045695 399.999998 360L399.999998 40C399.999998 28.954306 391.0456940000001 20 379.999998 20L19.999999 20C8.954304 20 -0.0000010000002 28.954306 -0.0000010000002 40L-0.0000010000002 360C-0.0000010000002 371.045695 8.954304 380 19.999999 380L379.999998 380zM360 300L40 300L40 60L360 60L360 300zM300 360L260 360L260 320L300 320L300 360zM360 360L320 360L320 320L360 320L360 360z" : "M300 300C311.045694 300 320 291.045695 320 280L320 40C320 28.954306 311.045694 20 300 20L20 20C8.954305 20 0 28.954306 0 40L0 280C0 291.045695 8.954305 300 20 300L300 300zM280 220L40 220L40 60L280 60L280 220zM220 280L180 280L180 240L220 240L220 280zM280 280L240 280L240 240L280 240L280 280z", fillRule: "evenodd", transform: `scale(0.05, -0.05) translate(${translation}, ${translation})`, style: style })));
});
Modal.defaultProps = {
    size: IconSize.STANDARD,
};
Modal.displayName = `Blueprint5.Icon.Modal`;
export default Modal;
//# sourceMappingURL=modal.js.map