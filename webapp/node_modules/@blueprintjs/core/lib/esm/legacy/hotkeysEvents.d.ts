/**
 * @fileoverview This component is DEPRECATED, and the code is frozen.
 * All changes & bugfixes should be made to HotkeysDialog2 instead.
 */
import { type ReactNode } from "react";
import { type HotkeyProps, type HotkeysProps } from "../components/hotkeys";
import { type KeyCombo } from "../components/hotkeys/hotkeyParser";
export declare enum HotkeyScope {
    LOCAL = "local",
    GLOBAL = "global"
}
export interface IHotkeyAction {
    combo: KeyCombo;
    props: HotkeyProps;
}
export declare class HotkeysEvents {
    private scope;
    private actions;
    constructor(scope: HotkeyScope);
    count(): number;
    clear(): void;
    setHotkeys(props: HotkeysProps & {
        children?: ReactNode;
    }): void;
    handleKeyDown: (e: KeyboardEvent) => void;
    handleKeyUp: (e: KeyboardEvent) => void;
    private invokeNamedCallbackIfComboRecognized;
    private isScope;
    private isTextInput;
}
