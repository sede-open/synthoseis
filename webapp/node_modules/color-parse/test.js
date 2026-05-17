import parse from './index.js'
import t from 'tape'

/** parse-color tests */

t('#ffa500', function (t) {
	t.deepEqual(parse('#ffa500'), {
		space: 'rgb',
		values: [255, 165, 0],
		alpha: 1
	});
	t.end()
});
t('#333', function (t) {
	t.deepEqual(parse('#333'), {
		space: 'rgb',
		values: [51, 51, 51],
		alpha: 1
	});
	t.end()
});
t('#f98', function (t) {
	t.deepEqual(parse('#f98'), {
		space: 'rgb',
		values: [255, 153, 136],
		alpha: 1
	});
	t.end()
});
t('lime', function (t) {
	t.deepEqual(parse('lime'), {
		space: 'rgb',
		values: [0, 255, 0],
		alpha: 1
	});
	t.deepEqual(parse('LIME'), {
		space: 'rgb',
		values: [0, 255, 0],
		alpha: 1
	});
	t.end()
});
t('hsl(210,50,50)', function (t) {
	t.deepEqual(parse('hsl(210,50,50)'), {
		space: 'hsl',
		values: [210, 50, 50],
		alpha: 1
	});
	t.end()
});
t('rgba(153,50,204,60%)', function (t) {
	t.deepEqual(parse('rgba(153,50,204,60%)'), {
		space: 'rgb',
		values: [153, 50, 204],
		alpha: 0.6
	});
	t.end()
});



t('#fef', function (t) {
	t.deepEqual(parse('#fef'), {
		space: 'rgb',
		values: [255, 238, 255],
		alpha: 1
	});
	t.end()
});
t('#fffFEF', function (t) {
	t.deepEqual(parse('#fffFEF'), {
		space: 'rgb',
		values: [255, 255, 239],
		alpha: 1
	});
	t.end()
});
t('rgb(244, 233, 100)', function (t) {
	t.deepEqual(parse('rgb(244, 233, 100)'), {
		space: 'rgb',
		values: [244, 233, 100],
		alpha: 1
	});
	t.end()
});
t('rgb(100%, 30%, 90%)', function (t) {
	t.deepEqual(parse('rgb(100%, 30%, 90%)'), {
		space: 'rgb',
		values: [255, 76.5, 229.5],
		alpha: 1
	});
	t.end()
});
t('transparent', function (t) {
	t.deepEqual(parse('transparent'), {
		space: 'rgb',
		values: [0, 0, 0],
		alpha: 0
	});
	t.end()
});
t('hsl(240, 100%, 50.5%)', function (t) {
	t.deepEqual(parse('hsl(240, 100%, 50.5%)'), {
		space: 'hsl',
		values: [240, 100, 50.5],
		alpha: 1
	});
	t.end()
});
t('hsl(240deg, 100%, 50.5%)', function (t) {
	t.deepEqual(parse('hsl(240deg, 100%, 50.5%)'), {
		space: 'hsl',
		values: [240, 100, 50.5],
		alpha: 1
	});
	t.end()
});
t('hwb(240, 100%, 50.5%)', function (t) {
	t.deepEqual(parse('hwb(240, 100%, 50.5%)'), {
		space: 'hwb',
		values: [240, 100, 50.5],
		alpha: 1
	});
	t.end()
});
t('hwb(240deg, 100%, 50.5%)', function (t) {
	t.deepEqual(parse('hwb(240deg, 100%, 50.5%)'), {
		space: 'hwb',
		values: [240, 100, 50.5],
		alpha: 1
	});
	t.end()
});
t('blue', function (t) {
	t.deepEqual(parse('blue'), {
		space: 'rgb',
		values: [0, 0, 255],
		alpha: 1
	});
	t.deepEqual(parse('BLUE'), {
		space: 'rgb',
		values: [0, 0, 255],
		alpha: 1
	});
	t.end()
});
t('rgb(244, 233, 100)', function (t) {
	t.deepEqual(parse('rgb(244, 233, 100)'), {
		space: 'rgb',
		values: [244, 233, 100],
		alpha: 1
	});
	t.end()
});
t('rgba(244, 233, 100, 0.5)', function (t) {
	t.deepEqual(parse('rgba(244, 233, 100, 0.5)'), {
		space: 'rgb',
		values: [244, 233, 100],
		alpha: 0.5
	});
	t.end()
});
t('hsla(244, 100%, 100%, 0.6)', function (t) {
	t.deepEqual(parse('hsla(244, 100%, 100%, 0.6)'), {
		space: 'hsl',
		values: [244, 100, 100],
		alpha: 0.6
	});
	t.end()
});
t('hwb(244, 100%, 100%, 0.6)', function (t) {
	t.deepEqual(parse('hwb(244, 100%, 100%, 0.6)'), {
		space: 'hwb',
		values: [244, 100, 100],
		alpha: 0.6
	});
	t.end()
});
t('hwb(244, 100%, 100%)', function (t) {
	t.deepEqual(parse('hwb(244, 100%, 100%)'), {
		space: 'hwb',
		values: [244, 100, 100],
		alpha: 1
	});
	t.end()
});
t('rgba(200, 20, 233, 0.2)', function (t) {
	t.deepEqual(parse('rgba(200, 20, 233, 0.2)'), {
		space: 'rgb',
		values: [200, 20, 233],
		alpha: 0.2
	});
	t.end()
});
t('rgba(200, 20, 233, 0)', function (t) {
	t.deepEqual(parse('rgba(200, 20, 233, 0)'), {
		space: 'rgb',
		values: [200, 20, 233],
		alpha: 0
	});
	t.end()
});
t('rgba(100%, 30%, 90%, 0.2)', function (t) {
	t.deepEqual(parse('rgba(100%, 30%, 90%, 0.2)'), {
		space: 'rgb',
		values: [255, 76.5, 229.5],
		alpha: 0.2
	});
	t.end()
});
t('rgba(200 20 233 / 0.2)', function (t) {
	t.deepEqual(parse('rgba(200, 20, 233, 0.2)'), {
		space: 'rgb',
		values: [200, 20, 233],
		alpha: 0.2
	});
	t.end()
});
t('rgba(200 20 233 / 20%)', function (t) {
	t.deepEqual(parse('rgba(200, 20, 233, 0.2)'), {
		space: 'rgb',
		values: [200, 20, 233],
		alpha: 0.2
	});
	t.end()
});
t('hsla(200, 20%, 33%, 0.2)', function (t) {
	t.deepEqual(parse('hsla(200, 20%, 33%, 0.2)'), {
		space: 'hsl',
		values: [200, 20, 33],
		alpha: 0.2
	});
	t.end()
});
t('hwb(200, 20%, 33%, 0.2)', function (t) {
	t.deepEqual(parse('hwb(200, 20%, 33%, 0.2)'), {
		space: 'hwb',
		values: [200, 20, 33],
		alpha: 0.2
	});
	t.end()
});
t('rgba(200, 20, 233, 0.2)', function (t) {
	t.deepEqual(parse('rgba(200, 20, 233, 0.2)'), {
		space: 'rgb',
		values: [200, 20, 233],
		alpha: 0.2
	});
	t.end()
});


t('rgba(300, 600, 100, 3)', function (t) {
	t.deepEqual(parse('rgba(300, 600, 100, 3)'), {
		space: 'rgb',
		values: [300, 600, 100],
		alpha: 3
	});
	t.end()
});
t('rgba(8000%, 100%, 333%, 88)', function (t) {
	t.deepEqual(parse('rgba(8000%, 100%, 333%, 88)'), {
		space: 'rgb',
		values: [20400, 255, 849.15],
		alpha: 88
	});
	t.end()
});
t('hsla(400, 10%, 200%, 10)', function (t) {
	t.deepEqual(parse('hsla(400, 10%, 200%, 10)'), {
		space: 'hsl',
		values: [400, 10, 200],
		alpha: 10
	});
	t.end()
});
t('hwb(400, 10%, 200%, 10)', function (t) {
	t.deepEqual(parse('hwb(400, 10%, 200%, 10)'), {
		space: 'hwb',
		values: [400, 10, 200],
		alpha: 10
	});
	t.end()
});
t('yellowblue', function (t) {
	t.deepEqual(parse('yellowblue'), { space: undefined, values: [], alpha: 1 });
	t.deepEqual(parse('YELLOWBLUE'), { space: undefined, values: [], alpha: 1 });
	t.end()
});


t('hsla(101.12, 45.2%, 21.0%, 1.0)', function (t) {
	t.deepEqual(parse('hsla(101.12, 45.2%, 21.0%, 1.0)'), {
		space: 'hsl',
		values: [101.12, 45.2, 21.0],
		alpha: 1
	});
	t.end()
});
t('hsla(101.12 45.2% 21.0% / 50%)', function (t) {
	t.deepEqual(parse('hsla(101.12 45.2% 21.0% / 50%)'), {
		space: 'hsl',
		values: [101.12, 45.2, 21.0],
		alpha: .5
	});
	t.end()
});
t('hsl(red, 10%, 10%)', function (t) {
	t.deepEqual(parse('hsl(red, 10%, 10%)'), {
		space: 'hsl',
		values: [0, 10, 10],
		alpha: 1
	});
	t.end()
});
t('hsl(red, 10%, 10%);', function (t) {
	t.deepEqual(parse('hsl(red, 10%, 10%);'), {
		space: 'hsl',
		values: [0, 10, 10],
		alpha: 1
	});
	t.end()
});
t('hsl(10deg, 10%, 10%)', function (t) {
	t.deepEqual(parse('hsl(10deg, 10%, 10%)'), {
		space: 'hsl',
		values: [10, 10, 10],
		alpha: 1
	});
	t.end()
});
t('hsl(1.5turn, 10%, 10%)', function (t) {
	t.deepEqual(parse('hsl(1.5turn, 10%, 10%)'), {
		space: 'hsl',
		values: [540, 10, 10],
		alpha: 1
	});
	t.end()
});
t('lch(5, 5, orange)', function (t) {
	t.deepEqual(parse('lch(5, 5, orange)'), {
		space: 'lch',
		values: [5, 5, 60],
		alpha: 1
	});
	t.end()
});
t('lch(5 5 orange / .5)', function (t) {
	t.deepEqual(parse('lch(5 5 orange / .5)'), {
		space: 'lch',
		values: [5, 5, 60],
		alpha: 0.5
	});
	t.end()
});
t('lab(0.25, 0.25, 0.25)', function (t) {
	t.deepEqual(parse('lab(0.25, 0.25, 0.25)'), {
		space: 'lab',
		values: [0.25, 0.25, 0.25],
		alpha: 1
	});
	t.end()
});
t('lab(0.25 0.25 0.25 / 0.5)', function (t) {
	t.deepEqual(parse('lab(0.25 0.25 0.25 / 0.5)'), {
		space: 'lab',
		values: [0.25, 0.25, 0.25],
		alpha: 0.5
	});
	t.end()
});

t('luv(0.25, 0.25, 0.25)', function (t) {
	t.deepEqual(parse('luv(0.25, 0.25, 0.25)'), {
		space: 'luv',
		values: [0.25, 0.25, 0.25],
		alpha: 1
	});
	t.end()
});
t('luv(0.25 0.25 0.25 / 0.5)', function (t) {
	t.deepEqual(parse('luv(0.25 0.25 0.25 / 0.5)'), {
		space: 'luv',
		values: [0.25, 0.25, 0.25],
		alpha: 0.5
	});
	t.end()
});

t('color(...)', function (t) {
	// --srgb: color(srgb 1 1 1);
	t.deepEqual(parse('color(srgb-linear 1 1 1)'), {
		space: 'srgb-linear',
		values: [1, 1, 1],
		alpha: 1
	});
	// --srgb-linear: color(srgb-linear 100% 100% 100% / 50%);
	t.deepEqual(parse('color(srgb-linear 100% 100% 100% / 50%)'), {
		space: 'srgb-linear',
		values: [1, 1, 1],
		alpha: 0.5
	});
	// --display-p3: color(display-p3 1 1 1);
	t.deepEqual(parse('color(display-p3 1 1 1)'), {
		space: 'display-p3',
		values: [1, 1, 1],
		alpha: 1
	});
	// --rec2020: color(rec2020 0 0 0);
	t.deepEqual(parse('color(rec2020 0 0 0)'), {
		space: 'rec2020',
		values: [0, 0, 0],
		alpha: 1
	});
	// --a98-rgb: color(a98-rgb 1 1 1 / 25%);
	t.deepEqual(parse('color(a98-rgb 1 1 1 / 25%)'), {
		space: 'a98-rgb',
		values: [1, 1, 1],
		alpha: 0.25
	});
	// --prophoto: color(prophoto-rgb 0% 0% 0%);
	t.deepEqual(parse('color(prophoto-rgb 0% 0% 0%)'), {
		space: 'prophoto-rgb',
		values: [0, 0, 0],
		alpha: 1
	});
	// --xyz: color(xyz 1 1 1);
	t.deepEqual(parse('color(xyz 1 1 1)'), {
		space: 'xyz',
		values: [1, 1, 1],
		alpha: 1
	});
	t.end()
});

t('oklab', function (t) {
	t.deepEqual(parse('oklab(40.1% 0.1143 0.045)'), {
		space: 'oklab',
		values: [0.401, 0.1143, 0.045],
		alpha: 1
	});
	t.deepEqual(parse('oklab(59.69% 0.1007 -0.1191 / 0.5)'), {
		space: 'oklab',
		values: [0.5969, 0.1007, -0.1191],
		alpha: 0.5
	});
	t.deepEqual(parse('oklab(0.123 100% -100% / 2)'), {
		space: 'oklab',
		values: [0.123, 0.4, -0.4],
		alpha: 2
	});
	t.deepEqual(parse('oklab(none none none / none)'), {
		space: 'oklab',
		values: [0, 0, 0],
		alpha: 0
	});
	t.end()
});
t('oklch', function (t) {
	t.deepEqual(parse('oklch(40.1% 0.1143 0.045)'), {
		space: 'oklch',
		values: [0.401, 0.1143, 0.045],
		alpha: 1
	});
	t.deepEqual(parse('oklch(59.69% 10% 49.77 / 0.5)'), {
		space: 'oklch',
		values: [0.5969, 0.04000000000000001, 49.77],
		alpha: 0.5
	});
	t.deepEqual(parse('oklch(40.1% 0.156 49.1% / .5)'), {
		space: 'oklch',
		values: [0.401, 0.156, 176.76],
		alpha: .5
	});
	t.deepEqual(parse('oklch(none none none / none)'), {
		space: 'oklch',
		values: [0, 0, 0],
		alpha: 0
	});
	t.end()
});

t('#afd6', function (t) {
	t.deepEqual(parse('#afd6'), {
		space: 'rgb',
		values: [170, 255, 221],
		alpha: 0.4
	});
	t.end()
});
t('#AFD6', function (t) {
	t.deepEqual(parse('#afd6'), {
		space: 'rgb',
		values: [170, 255, 221],
		alpha: 0.4
	});
	t.end()
});
t('#aaffdd66', function (t) {
	t.deepEqual(parse('#aaffdd66'), {
		space: 'rgb',
		values: [170, 255, 221],
		alpha: 0.4
	});
	t.end()
});
t('#AAFFDD66', function (t) {
	t.deepEqual(parse('#AAFFDD66'), {
		space: 'rgb',
		values: [170, 255, 221],
		alpha: 0.4
	});
	t.end()
});
t('(R12 / G45 / B234)', function (t) {
	t.deepEqual(parse('(R12 / G45 / B234)'), {
		space: 'rgb',
		values: [12, 45, 234],
		alpha: 1
	});
	t.end()
});
t('R:12 G:45 B:234', function (t) {
	t.deepEqual(parse('R:12 G:45 B:234'), {
		space: 'rgb',
		values: [12, 45, 234],
		alpha: 1
	});
	t.end()
});
t('C100/M80/Y0/K35', function (t) {
	t.deepEqual(parse('C100/M80/Y0/K35'), {
		space: 'cmyk',
		values: [100, 80, 0, 35],
		alpha: 1
	});
	t.end()
});
t.skip('Array', function (t) {
	t.deepEqual(parse([1, 2, 3]), {
		space: 'rgb',
		values: [1, 2, 3],
		alpha: 1
	});
	t.end()
});
t.skip('Object', function (t) {
	t.deepEqual(parse({ r: 1, g: 2, b: 3 }), {
		space: 'rgb',
		values: [1, 2, 3],
		alpha: 1
	});
	t.deepEqual(parse({ red: 1, green: 2, blue: 3 }), {
		space: 'rgb',
		values: [1, 2, 3],
		alpha: 1
	});
	t.deepEqual(parse({ h: 1, s: 2, l: 3 }), {
		space: 'hsl',
		values: [1, 2, 3],
		alpha: 1
	});
	t.end()
});
t('Number', function (t) {
	t.deepEqual(parse(0xA141E), {
		space: 'rgb',
		values: [10, 20, 30],
		alpha: 1
	});
	t.deepEqual(parse(0xff), {
		space: 'rgb',
		values: [0x00, 0x00, 0xff],
		alpha: 1
	});
	t.deepEqual(parse(0xff0000), {
		space: 'rgb',
		values: [0xff, 0x00, 0x00],
		alpha: 1
	});
	t.deepEqual(parse(0x0000ff), {
		space: 'rgb',
		values: [0x00, 0x00, 0xff],
		alpha: 1
	});
	// t.deepEqual(parse(new Number(0x0000ff)), {
	// 	space: 'rgb',
	// 	values: [0x00, 0x00, 0xff],
	// 	alpha: 1
	// });
	t.end()
});
