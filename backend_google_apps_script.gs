/**
 * Free backend for the risk-attitude perception study (no server needed).
 * Saves each submission as a row in a Google Sheet.
 *
 * SETUP (5 minutes):
 *  1. Create a new Google Sheet.
 *  2. Extensions  ->  Apps Script. Delete the default code, paste THIS file, Save.
 *  3. Deploy  ->  New deployment  ->  type "Web app".
 *       - Execute as:  Me
 *       - Who has access:  Anyone
 *     Click Deploy, authorize, and COPY the "Web app URL".
 *  4. In index.html set  CONFIG.dataEndpoint = "<that Web app URL>".
 *  5. Test: open index.html, finish a session; a row should appear in the sheet.
 *
 * Each row stores summary columns + the full JSON (so nothing is lost).
 * The full JSON column lets you reconstruct everything in analysis.
 */

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(30000);                       // avoid concurrent-write collisions
  try {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = ss.getSheetByName('data') || ss.insertSheet('data');
    if (sheet.getLastRow() === 0) {
      sheet.appendRow(['received_at', 'subjectId', 'prolific_pid', 'study_id', 'session_id',
                       'flagged', 'n_comparisons', 'attn_passed', 'attn_total',
                       'dospert_overall', 'json']);
    }
    var body = (e && e.postData && e.postData.contents) ? e.postData.contents : '{}';
    var d = {};
    try { d = JSON.parse(body); } catch (err) {}
    var q = d.quality || {}, p = d.prolific || {}, dem = d.demographics || {},
        dos = dem.dospert || {};
    sheet.appendRow([
      new Date(), d.subjectId || '', p.pid || '', p.study || '', p.session || '',
      q.flagged === true, d.n_comparisons || '',
      q.attentionPassed != null ? q.attentionPassed : '',
      q.attentionTotal != null ? q.attentionTotal : '',
      dos.overall != null ? dos.overall : '', body
    ]);
    return ContentService.createTextOutput(JSON.stringify({ ok: true }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ ok: false, error: String(err) }))
      .setMimeType(ContentService.MimeType.JSON);
  } finally {
    lock.releaseLock();
  }
}

function doGet() {                            // health check
  return ContentService.createTextOutput('ok');
}
