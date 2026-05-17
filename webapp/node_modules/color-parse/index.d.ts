export default function parse(color: string): {
  space: 'rgb' | 'hsl' | 'hwb' | 'cmyk' | 'lab' | 'lch' | 'oklab' | 'oklch' | string;
  values: number[];
  alpha: number;
}
