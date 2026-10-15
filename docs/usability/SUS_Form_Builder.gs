/**
 * Builds the usability-evaluation form (thesis Section 5.8, Appendix L) in Google Forms, with a linked
 * response spreadsheet, and scores the System Usability Scale (SUS).
 *
 * HOW TO USE
 *  1. Go to https://script.google.com  ->  New project. Name it "SUS form".
 *  2. Delete the sample code, paste this whole file, click Save.
 *  3. Choose the function "createSusForm" in the toolbar and click Run.
 *     Approve the permission request (Google will ask once - it only creates a form and a sheet in YOUR Drive).
 *  4. Open the Execution log: it prints the EDIT link, the PUBLIC link and the RESPONSES SHEET link.
 *  5. Open the PUBLIC link and submit one test response with code P00. Then delete it:
 *     form editor -> Responses -> three dots -> "Delete all responses" (or delete just that one).
 *  6. After the sessions, run "scoreResponses": it writes a "SUS scores" tab in the responses sheet
 *     (each participant's score, then n, mean, 95% CI, SD, median, minimum, maximum and the benchmark of 68).
 *
 * RULES THAT KEEP THE SUS VALID (do not edit these parts of the form)
 *  - The ten statements, their order and the 1-5 scale are the standard SUS (originally developed by Brooke), word for word as in
 *    Appendix L.3. Do not reword, reorder, translate or add questions between them.
 *  - Question titles start with their code (SUS1. ... SUS10.). Do not rename them: the scoring matches on them.
 *  - No names or emails are collected. The participant code (P01, P02 ...) links the form to your task sheet.
 */
var RESEARCHER = 'M. S. F. Shazna (COBSCCOMP242P-019@student.nibm.lk)';
var SUPERVISOR = 'Ms. L. Sumudu Chathurika (sumudu@nibm.lk), School of Computing and Engineering, NIBM';
var BENCHMARK = 68;

var SUS_ITEMS = [
  'I think that I would like to use this system frequently.',
  'I found the system unnecessarily complex.',
  'I thought the system was easy to use.',
  'I think that I would need the support of a technical person to be able to use this system.',
  'I found the various functions in this system were well integrated.',
  'I thought there was too much inconsistency in this system.',
  'I would imagine that most people would learn to use this system very quickly.',
  'I found the system very cumbersome to use.',
  'I felt very confident using the system.',
  'I needed to learn a lot of things before I could get going with this system.'
];

var ELIGIBLE = 'All of these apply to me';
var NOT_ELIGIBLE = 'One or more of these does NOT apply to me';

function createSusForm() {
  var form = FormApp.create('Usability Evaluation – Time to First IT Job Website');
  form.setDescription(
      'This short questionnaire is part of my final-year BSc (Hons) Computing research at NIBM. You have just been ' +
      'asked to try a research website that estimates how long Sri Lankan IT graduates take to find their first IT ' +
      'job. The questionnaire asks how easy the website was to use.\n' +
      'It is the website that is being evaluated, not you: there are no right or wrong answers.');
  form.setCollectEmail(false);
  form.setLimitOneResponsePerUser(false);      // several participants answer on the researcher's laptop
  form.setAllowResponseEdits(false);
  form.setProgressBar(true);
  form.setShuffleQuestions(false);             // the SUS must keep its standard order
  form.setShowLinkToRespondAgain(true);        // "Submit another response" for the next participant
  form.setPublishingSummary(false);            // participants cannot see other people's answers
  try { form.setRequireLogin(false); } catch (err) { /* only exists for Google Workspace accounts */ }
  form.setConfirmationMessage(
      'Thank you for taking part. Your answers have been recorded under your participant code only.\n' +
      'Please hand the laptop back to the researcher.');

  // ---------------------------------------------------------------- Page 1: information, consent, code
  form.addSectionHeaderItem().setTitle('Participant information').setHelpText(
      '• What happens: you use the website for about 15 minutes to complete five short tasks while the researcher ' +
      'notes where it is easy or difficult, then you answer this questionnaire (about 5 minutes).\n' +
      '• Voluntary: taking part is entirely voluntary. You may stop at any time without giving a reason.\n' +
      '• No personal details: your name, email and the profile details you type into the website are NOT recorded. ' +
      'You are identified only by a participant code (for example P03). Because the code is not linked to your name, ' +
      'your answers cannot be identified or withdrawn once submitted.\n' +
      '• Data use: answers are used only for academic research, stored securely on password-protected devices, never ' +
      'shared with third parties, and deleted when the study is completed, in line with Sri Lanka\'s Personal Data ' +
      'Protection Act, No. 9 of 2022.\n' +
      '• Contact: Researcher: ' + RESEARCHER + '. Supervisor: ' + SUPERVISOR + '.');

  var consentItems = [
    'I have read and understood the information above.',
    'I understand that taking part is voluntary and that I can stop at any time without giving a reason.',
    'I understand that my name and personal details are not recorded, only a participant code.',
    'I agree to take part in this usability evaluation.'
  ];
  form.addCheckboxItem().setTitle('Consent').setChoiceValues(consentItems).setRequired(true)
      .setValidation(FormApp.createCheckboxValidation().requireSelectAtLeast(consentItems.length)
          .setHelpText('Please tick all four boxes to take part.').build());

  form.addTextItem().setTitle('Participant code')
      .setHelpText('The researcher will tell you your code (P01, P02 ...).').setRequired(true)
      .setValidation(FormApp.createTextValidation().requireTextMatchesPattern('^P[0-9]{2}$')
          .setHelpText('Use a capital P followed by two digits, for example P03.').build());

  var eligibility = form.addMultipleChoiceItem().setTitle('Eligibility').setRequired(true).setHelpText(
      'Please confirm ALL of the following:\n' +
      '• I am a final-year IT / computing undergraduate, OR I have completed an IT / computing degree.\n' +
      '• I did NOT fill in the graduate questionnaire for this research ("Graduate Employability Survey – Time to ' +
      'First IT Job in Sri Lanka").\n' +
      '• I have not used this website before today and did not help to build or test it.\n' +
      '• I am 18 years or older.');

  // ---------------------------------------------------------------- Page 2: about you
  var aboutPage = form.addPageBreakItem().setTitle('About you')
      .setHelpText('A few background questions so the results can be described. No names are collected.');
  form.addMultipleChoiceItem().setTitle('B1. Which describes you?').setRequired(true)
      .setChoiceValues(['Final-year undergraduate', 'Graduated, still looking for my first IT job',
                        'Graduated, already working in IT', 'Graduated, working outside IT']);
  form.addMultipleChoiceItem().setTitle('B2. Degree area').setRequired(true)
      .setChoiceValues(['Computer Science', 'Information Technology', 'Software Engineering', 'Information Systems',
                        'Cyber Security', 'Data Science / AI', 'Other computing degree']);
  form.addMultipleChoiceItem().setTitle('B3. Type of institution').setRequired(true)
      .setChoiceValues(['State university', 'Non-state institute', 'Foreign-affiliated institute']);
  form.addMultipleChoiceItem().setTitle('B4. How often do you use career or job-search websites or apps?')
      .setRequired(true).setChoiceValues(['Never', 'A few times a year', 'Monthly', 'Weekly or more']);

  // ---------------------------------------------------------------- Page 3: pause for the tasks
  form.addPageBreakItem().setTitle('Please stop here').setHelpText(
      'Tell the researcher you are ready, and complete the five tasks on the website.\n\n' +
      'Click "Next" only AFTER you have finished all five tasks.');

  // ---------------------------------------------------------------- Page 4: SUS (standard, unmodified)
  form.addPageBreakItem().setTitle('Your experience of the website (System Usability Scale)').setHelpText(
      'For each statement, choose how much you agree, thinking about the website you have just used ("the system"). ' +
      'Answer every statement, quickly, with your first reaction. If you are not sure, choose the middle point (3).');
  SUS_ITEMS.forEach(function (text, i) {
    form.addScaleItem().setTitle('SUS' + (i + 1) + '. ' + text).setBounds(1, 5)
        .setLabels('Strongly disagree', 'Strongly agree').setRequired(true);
  });

  // ---------------------------------------------------------------- Page 5: feedback
  form.addPageBreakItem().setTitle('Your comments').setHelpText('Optional, but very helpful.');
  form.addMultipleChoiceItem().setTitle('C1. Which profile did you enter in the tasks?').setRequired(true)
      .setChoiceValues(['My own details', 'The example profile card', 'A mix of both']);
  form.addParagraphTextItem().setTitle('C2. What was the most useful part of the website?');
  form.addParagraphTextItem().setTitle('C3. What was confusing, difficult or should be improved?');
  form.addParagraphTextItem().setTitle('C4. Anything else you would like to say?');

  // ---------------------------------------------------------------- Page 6: not eligible (skipped by eligible people)
  var notEligiblePage = form.addPageBreakItem().setTitle('Thank you').setHelpText(
      'Thank you for your interest. This evaluation is only for people who meet all the eligibility conditions, so ' +
      'please click "Submit" and let the researcher know. Your response will not be used.');
  notEligiblePage.setGoToPage(FormApp.PageNavigationType.SUBMIT);    // the comments page submits instead of coming here

  eligibility.setChoices([eligibility.createChoice(ELIGIBLE, aboutPage),
                          eligibility.createChoice(NOT_ELIGIBLE, notEligiblePage)]);

  // ---------------------------------------------------------------- linked responses sheet
  var ss = SpreadsheetApp.create('Usability Evaluation – responses');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, ss.getId());
  PropertiesService.getScriptProperties().setProperties({ formId: form.getId(), sheetId: ss.getId() });

  Logger.log('EDIT link:      ' + form.getEditUrl());
  Logger.log('PUBLIC link:    ' + form.getPublishedUrl());
  Logger.log('RESPONSES sheet: ' + ss.getUrl());
  Logger.log('Next: submit one test response as P00, check it, then delete it before the first real session.');
}

/** Scores every eligible, complete response and writes a "SUS scores" tab in the responses sheet. */
function scoreResponses() {
  var props = PropertiesService.getScriptProperties();
  var form = FormApp.openById(props.getProperty('formId'));
  var ss = SpreadsheetApp.openById(props.getProperty('sheetId'));

  var rows = [], skipped = [], seen = {};
  form.getResponses().forEach(function (resp) {
    var a = {};
    resp.getItemResponses().forEach(function (ir) { a[ir.getItem().getTitle()] = ir.getResponse(); });
    var code = a['Participant code'];
    if (code === 'P00') return;                                            // test response
    if (a['Eligibility'] !== ELIGIBLE) { skipped.push(code + ' (not eligible)'); return; }
    var ratings = SUS_ITEMS.map(function (_, i) {
      var key = Object.keys(a).filter(function (k) { return k.indexOf('SUS' + (i + 1) + '. ') === 0; })[0];
      return key ? Number(a[key]) : NaN;
    });
    if (ratings.some(isNaN)) { skipped.push(code + ' (SUS incomplete)'); return; }
    if (seen[code]) { skipped.push(code + ' (duplicate code - check which one is right)'); }
    seen[code] = true;
    // Odd items: rating - 1. Even items: 5 - rating. Sum x 2.5 = 0-100.
    var score = 2.5 * ratings.reduce(function (s, r, i) { return s + (i % 2 === 0 ? r - 1 : 5 - r); }, 0);
    rows.push([code, resp.getTimestamp()].concat(ratings, [score]));
  });

  var sheet = ss.getSheetByName('SUS scores') || ss.insertSheet('SUS scores');
  sheet.clear();
  var header = ['Participant', 'Submitted'].concat(SUS_ITEMS.map(function (_, i) { return 'SUS' + (i + 1); }),
                                                   ['SUS score (0-100)']);
  sheet.getRange(1, 1, 1, header.length).setValues([header]).setFontWeight('bold');
  if (rows.length) sheet.getRange(2, 1, rows.length, header.length).setValues(rows);

  var scores = rows.map(function (r) { return r[r.length - 1]; });
  var n = scores.length, summary = [['Participants scored', n]];
  if (n) {
    var mean = scores.reduce(function (s, x) { return s + x; }, 0) / n;
    var sd = n > 1 ? Math.sqrt(scores.reduce(function (s, x) { return s + Math.pow(x - mean, 2); }, 0) / (n - 1)) : 0;
    var half = n > 1 ? tCritical_(n - 1) * sd / Math.sqrt(n) : NaN;
    var sorted = scores.slice().sort(function (x, y) { return x - y; });
    var median = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    summary = summary.concat([
      ['Mean SUS score', round_(mean)],
      ['95% CI (t distribution)', n > 1 ? round_(mean - half) + ' to ' + round_(mean + half) : 'needs 2+ participants'],
      ['Standard deviation', round_(sd)],
      ['Median', round_(median)],
      ['Minimum', sorted[0]],
      ['Maximum', sorted[n - 1]],
      ['Benchmark', BENCHMARK],
      ['Mean at or above the benchmark?', mean >= BENCHMARK ? 'Yes' : 'No'],
      ['Target sample (8-12) met?', n >= 8 && n <= 12 ? 'Yes' : 'No (' + n + ')']
    ]);
  }
  if (skipped.length) summary.push(['Not scored', skipped.join('; ')]);
  var top = rows.length + 3;
  sheet.getRange(top, 1, summary.length, 2).setValues(summary);
  sheet.getRange(top, 1, summary.length, 1).setFontWeight('bold');
  sheet.autoResizeColumns(1, header.length);
  Logger.log('Scored ' + n + ' participants. Open the "SUS scores" tab: ' + ss.getUrl());
}

function tCritical_(df) {          // two-sided 95% critical values of Student's t
  var t = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
           2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
           2.042];
  return df <= 30 ? t[df - 1] : 1.96;
}

function round_(x) { return Math.round(x * 10) / 10; }
