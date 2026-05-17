import { type DialogProps, type HotkeyProps } from "../components";
interface HotkeysDialogProps extends DialogProps {
    /**
     * This string displayed as the group name in the hotkeys dialog for all
     * global hotkeys.
     */
    globalHotkeysGroup?: string;
}
/** @deprecated use HotkeysProvider */
export declare function isHotkeysDialogShowing(): boolean;
/** @deprecated use HotkeysProvider */
export declare function setHotkeysDialogProps(props: Partial<HotkeysDialogProps>): void;
/** @deprecated use HotkeysProvider */
export declare function showHotkeysDialog(hotkeys: HotkeyProps[]): void;
/** @deprecated use HotkeysProvider */
export declare function hideHotkeysDialog(): void;
/**
 * Use this function instead of `hideHotkeysDialog` if you need to ensure that all hotkey listeners
 * have time to execute with the dialog in a consistent open state. This can avoid flickering the
 * dialog between open and closed states as successive listeners fire.
 *
 * @deprecated use HotkeysProvider
 */
export declare function hideHotkeysDialogAfterDelay(): void;
export {};
