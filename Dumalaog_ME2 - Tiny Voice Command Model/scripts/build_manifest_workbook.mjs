import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const [payloadPath, output] = process.argv.slice(2);
const data = JSON.parse(await fs.readFile(payloadPath, 'utf8'));
const wb = Workbook.create();
const sheets = [];
function makeSheet(name, subtitle, headers, rows, widths) {
  const sh = wb.worksheets.add(name);
  sheets.push(sh);
  sh.showGridLines = false;
  const last = String.fromCharCode(64 + headers.length);
  sh.getRange('A1').values = [[data.title]];
  sh.getRange('A2').values = [[subtitle]];
  sh.getRange('A3').values = [[`Exported ${data.created_at}; active pair ${data.active_run}`]];
  sh.getRange(`A5:${last}${5 + rows.length}`).values = [headers, ...rows];
  const table = sh.tables.add(`A5:${last}${5 + rows.length}`, true, name.replaceAll(' ', '') + 'Table');
  table.style = 'TableStyleMedium2';
  sh.getRange(`A1:${last}${5 + rows.length}`).format.font = {name:'Arial', size:11, color:'#202020'};
  sh.getRange('A1').format.font = {name:'Arial', size:15, bold:true};
  sh.getRange('A2:A3').format.font = {name:'Arial', size:10, color:'#555555'};
  sh.getRange(`A5:${last}5`).format = {fill:'#26374B', font:{name:'Arial', size:11, bold:true, color:'#FFFFFF'}, wrapText:true, rowHeight:36};
  sh.getRange(`A6:${last}${5 + rows.length}`).format = {wrapText:true, rowHeight:38, verticalAlignment:'center'};
  sh.getRange(`A1:${last}3`).format.rowHeight = 22;
  for (let i=0; i<widths.length; i++) {
    sh.getRange(`${String.fromCharCode(65+i)}1:${String.fromCharCode(65+i)}${5+rows.length}`).format.columnWidth = widths[i];
  }
  sh.freezePanes.freezeRows(5);
  sh.freezePanes.freezeColumns(1);
  return sh;
}
makeSheet('Command phrases', 'Canonical prompts come from the same files used by the recorder. Spoken transcripts are not inferred.',
 ['Label', 'Canonical suggested phrase', 'Training role', 'Matching reference transcripts', 'Raw saved takes', 'Unique clips in active run', 'Ground truth source'],
 data.commands, [33,52,20,21,16,22,54]);
makeSheet('Recordings', `${data.raw_rows} saved rows; ${data.logged_prompt_rows} logged prompts. Blank historical prompts mean unknown. Current phrase is a reference, not a recovered transcript.`,
 ['Row', 'Original label', 'Mapped intent', 'Current canonical phrase', 'Prompt logged at save', 'Prompt variant index', 'Prompt provenance', 'Spoken transcript status', 'Speaker ID', 'Condition', 'Active run split / status', 'Duplicate group size', 'Source ID', 'Relative WAV path', 'PCM SHA-256'],
 data.recordings, [7,31,31,48,48,18,28,29,16,17,30,19,38,65,72]);
const perf = makeSheet('Intent performance', `Reference accuracy ${(100*data.reference_accuracy).toFixed(2)}%; personal accuracy ${(100*data.personal_accuracy).toFixed(2)}%. Reused reference test; personal speakers overlap splits.`,
 ['Intent', 'Exact recorder phrase', 'Reference recall', 'Reference F1', 'Reference test n', 'Personal precision', 'Personal recall', 'Personal F1', 'Personal test n', 'Recording priority', 'Live confidence gate', 'Live margin gate', 'Evaluation policy'],
 data.performance, [32,50,18,18,18,18,18,18,18,24,20,20,33]);
perf.getRange('C6:D36').format.numberFormat='0.00%';
perf.getRange('F6:H36').format.numberFormat='0.00%';
perf.getRange('K6:L36').format.numberFormat='0%';
perf.getRange('C6:D36').conditionalFormats.add('cellIs', {operator:'lessThan', formula:0.95, format:{fill:'#FFF0C2'}});
perf.getRange('F6:H36').conditionalFormats.add('cellIs', {operator:'lessThan', formula:0.95, format:{fill:'#FFF0C2'}});
const wake = makeSheet('Wake performance', `${data.snapshot_rows}-row frozen training snapshot. Five placements per held-out personal clip; reference negatives reused. Cutoff is confidence, not accuracy.`,
 ['Operating point', 'Wake cutoff', 'Wake hits', 'Wake test n', 'Wake recall', 'Wake precision', 'Wake F1', 'Binary accuracy', 'Personal false accepts', 'Personal negatives n', 'Reference false accepts', 'Reference negatives n'],
 data.wake, [49,19,14,15,18,18,18,20,23,22,24,23]);
wake.getRange('B6:B7').format.numberFormat='0.000000%';
wake.getRange('E6:H7').format.numberFormat='0.00%';
wake.getRange('A10').values=[['Intent weight origin']];
wake.getRange('A11').values=[[`${data.provenance.intent?.training || 'from_random_initialization'}; source ${data.provenance.intent?.source_run || data.active_run}`]];
wake.getRange('A13').values=[['Historical prompts remain unknown. No speech-to-text or manual audio transcription was performed.']];
for (const sh of sheets) {
  const preview = await wb.render({sheetName:sh.name, range:sh.name==='Recordings'?'A1:G13':(sh.name==='Wake performance'?'A1:H13':'A1:F14'), scale:1, format:'png'});
  await fs.writeFile(path.join(output, sh.name.replaceAll(' ', '_')+'.png'), new Uint8Array(await preview.arrayBuffer()));
}
const inspected = await wb.inspect({kind:'workbook,sheet,table', maxChars:3500, tableMaxRows:2, tableMaxCols:3});
await fs.writeFile(path.join(output,'inspection.json'),JSON.stringify(inspected,null,2));
const errors = await wb.inspect({kind:'match', searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A', options:{useRegex:true,maxResults:20},maxChars:1000});
await fs.writeFile(path.join(output,'error_scan.json'),JSON.stringify(errors,null,2));
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(output,'manifest.xlsx'));
console.log(JSON.stringify({sheets:sheets.map(s=>s.name), recordingRows:data.recordings.length, output:path.join(output,'manifest.xlsx')}));
