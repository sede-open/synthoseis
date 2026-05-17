/*
 * World Calendars
 * https://github.com/alexcjohnson/world-calendars
 *
 * Batch-converted from kbwood/calendars
 * Many thanks to Keith Wood and all of the contributors to the original project!
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

﻿/* http://keith-wood.name/calendars.html
   Persian calendar for jQuery v2.0.2.
   Written by Keith Wood (wood.keith{at}optusnet.com.au) August 2009.
   Available under the MIT (http://keith-wood.name/licence.html) license.
   Please attribute the author if you use it. */

var main = require('../main');
var assign = require('object-assign');

/** Implementation of the Persian or Jalali calendar.

       Modified Keith Wood's code by Mojtaba Samimi 2025.
    Persian Calendar is a calendar based on solar system.
    In the code below mean tropical year is used to calculate leap years.
    You may also refer to the figure 2 presented on page 15 of the book titled
    Intelligent Design using Solar-Climatic Vision.
    Free download links:
    https://depositonce.tu-berlin.de/items/c091139a-09cf-44c3-99a9-6adf59f7eaf8
    https://depositonce.tu-berlin.de/bitstreams/25a20942-2433-4ebe-90e2-0dfa69df5563/download

    See also <a href="http://en.wikipedia.org/wiki/Iranian_calendar">http://en.wikipedia.org/wiki/Iranian_calendar</a>.
    @class PersianCalendar
    @param [language=''] {string} The language code (default English) for localisation. */
function PersianCalendar(language) {
    this.local = this.regionalOptions[language || ''] || this.regionalOptions[''];
}

function _leapYear(year) {
    // 475 S.H. (1096 A.D.) possibly when the solar calendar is adjusted by
    // Omar Khayyam (https://en.wikipedia.org/wiki/Omar_Khayyam)
    var x = year - 475;
    if(year < 0) x++;

    // diff between approximate tropical year and 365
    var c = 0.242197;

    var v0 = c * x;
    var v1 = c * (x + 1);

    var r0 = v0 - Math.floor(v0);
    var r1 = v1 - Math.floor(v1);

    return r0 > r1;
}

PersianCalendar.prototype = new main.baseCalendar;

assign(PersianCalendar.prototype, {
    /** The calendar name.
        @memberof PersianCalendar */
    name: 'Persian',
    /** Julian date of start of Persian epoch: 19 March 622 CE.
        @memberof PersianCalendar */
    jdEpoch: 1948320.5,
    /** Days per month in a common year.
        @memberof PersianCalendar */
    daysPerMonth: [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29],
    /** <code>true</code> if has a year zero, <code>false</code> if not.
        @memberof PersianCalendar */
    hasYearZero: false,
    /** The minimum month number.
        @memberof PersianCalendar */
    minMonth: 1,
    /** The first month in the year.
        @memberof PersianCalendar */
    firstMonth: 1,
    /** The minimum day number.
        @memberof PersianCalendar */
    minDay: 1,

    /** Localisations for the plugin.
        Entries are objects indexed by the language code ('' being the default US/English).
        Each object has the following attributes.
        @memberof PersianCalendar
        @property name {string} The calendar name.
        @property epochs {string[]} The epoch names.
        @property monthNames {string[]} The long names of the months of the year.
        @property monthNamesShort {string[]} The short names of the months of the year.
        @property dayNames {string[]} The long names of the days of the week.
        @property dayNamesShort {string[]} The short names of the days of the week.
        @property dayNamesMin {string[]} The minimal names of the days of the week.
        @property dateFormat {string} The date format for this calendar.
                See the options on <a href="BaseCalendar.html#formatDate"><code>formatDate</code></a> for details.
        @property firstDay {number} The number of the first day of the week, starting at 0.
        @property isRTL {number} <code>true</code> if this localisation reads right-to-left. */
    regionalOptions: { // Localisations
        '': {
            name: 'Persian',
            epochs: ['BP', 'AP'],
            monthNames: ['Farvardin', 'Ordibehesht', 'Khordad', 'Tir', 'Mordad', 'Shahrivar',
            'Mehr', 'Aban', 'Azar', 'Dey', 'Bahman', 'Esfand'],
            monthNamesShort: ['Far', 'Ord', 'Kho', 'Tir', 'Mor', 'Sha', 'Meh', 'Aba', 'Aza', 'Dey', 'Bah', 'Esf'],
            dayNames: ['Yekshanbeh', 'Doshanbeh', 'Seshanbeh', 'Chahārshanbeh', 'Panjshanbeh', 'Jom\'eh', 'Shanbeh'],
            dayNamesShort: ['Yek', 'Do', 'Se', 'Cha', 'Panj', 'Jom', 'Sha'],
            dayNamesMin: ['Ye','Do','Se','Ch','Pa','Jo','Sh'],
            digits: null,
            dateFormat: 'yyyy/mm/dd',
            firstDay: 6,
            isRTL: false
        }
    },

    /** Determine whether this date is in a leap year.
        @memberof PersianCalendar
        @param year {CDate|number} The date to examine or the year to examine.
        @return {boolean} <code>true</code> if this is a leap year, <code>false</code> if not.
        @throws Error if an invalid year or a different calendar used. */
    leapYear: function(year) {
        var date = this._validate(year, this.minMonth, this.minDay, main.local.invalidYear);

        return _leapYear(date.year());
    },

    /** Determine the week of the year for a date.
        @memberof PersianCalendar
        @param year {CDate|number} The date to examine or the year to examine.
        @param [month] {number} The month to examine.
        @param [day] {number} The day to examine.
        @return {number} The week of the year.
        @throws Error if an invalid date or a different calendar used. */
    weekOfYear: function(year, month, day) {
        // Find Saturday of this week starting on Saturday
        var checkDate = this.newDate(year, month, day);
        checkDate.add(-((checkDate.dayOfWeek() + 1) % 7), 'd');
        return Math.floor((checkDate.dayOfYear() - 1) / 7) + 1;
    },

    /** Retrieve the number of days in a month.
        @memberof PersianCalendar
        @param year {CDate|number} The date to examine or the year of the month.
        @param [month] {number} The month.
        @return {number} The number of days in this month.
        @throws Error if an invalid month/year or a different calendar used. */
    daysInMonth: function(year, month) {
        var date = this._validate(year, month, this.minDay, main.local.invalidMonth);
        return this.daysPerMonth[date.month() - 1] +
            (date.month() === 12 && this.leapYear(date.year()) ? 1 : 0);
    },

    /** Determine whether this date is a week day.
        @memberof PersianCalendar
        @param year {CDate|number} The date to examine or the year to examine.
        @param [month] {number} The month to examine.
        @param [day] {number} The day to examine.
        @return {boolean} <code>true</code> if a week day, <code>false</code> if not.
        @throws Error if an invalid date or a different calendar used. */
    weekDay: function(year, month, day) {
        return this.dayOfWeek(year, month, day) !== 5;
    },

    /** Retrieve the Julian date equivalent for this date,
        i.e. days since January 1, 4713 BCE Greenwich noon.
        @memberof PersianCalendar
        @param year {CDate|number} The date to convert or the year to convert.
        @param [month] {number} The month to convert.
        @param [day] {number} The day to convert.
        @return {number} The equivalent Julian date.
        @throws Error if an invalid date or a different calendar used. */
    toJD: function(year, month, day) {
        var date = this._validate(year, month, day, main.local.invalidDate);
        year = date.year();
        month = date.month();
        day = date.day();

        var nLeapYearsSince = 0;
        if(year > 0) {
            for(var i = 1; i < year; i++) {
                if(_leapYear(i)) nLeapYearsSince++;
            }
        } else if(year < 0) {
            for(var i = year; i < 0; i++) {
                if(_leapYear(i)) nLeapYearsSince--;
            }
        }

        return day + (month <= 7 ? (month - 1) * 31 : (month - 1) * 30 + 6) +
            (year > 0 ? year - 1 : year) * 365 + nLeapYearsSince + this.jdEpoch - 1;
    },

    /** Create a new date from a Julian date.
        @memberof PersianCalendar
        @param jd {number} The Julian date to convert.
        @return {CDate} The equivalent date. */
    fromJD: function(jd) {
        jd = Math.floor(jd) + 0.5;

        // find year
        var y = 475 + (jd - this.toJD(475, 1, 1)) / 365.242197;
        var year = Math.floor(y);
        if(year <= 0) year--;

        if(jd > this.toJD(year, 12, _leapYear(year) ?  30 : 29)) {
            year++;
            if(year === 0) year++;
        }

        var yday = jd - this.toJD(year, 1, 1) + 1;
        var month = (yday <= 186 ? Math.ceil(yday / 31) : Math.ceil((yday - 6) / 30));
        var day = jd - this.toJD(year, month, 1) + 1;

        return this.newDate(year, month, day);
    }
});

// Persian (Jalali) calendar implementation
main.calendars.persian = PersianCalendar;
main.calendars.jalali = PersianCalendar;

