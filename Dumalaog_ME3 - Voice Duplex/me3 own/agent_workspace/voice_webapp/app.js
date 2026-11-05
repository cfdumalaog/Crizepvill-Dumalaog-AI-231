const statusText = document.querySelector('#statusText');
const lastCommand = document.querySelector('#lastCommand');
const activityLog = document.querySelector('#activityLog');
const form = document.querySelector('#commandForm');
const input = document.querySelector('#commandInput');
const micButton = document.querySelector('#micBtn');

function addActivity(message) {
  const item = document.createElement('li');
  item.textContent = message;
  activityLog.prepend(item);
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const command = input.value.trim();
  if (!command) return;
  statusText.textContent = 'Processing';
  lastCommand.textContent = command;
  addActivity(`tool call queued: ${command}`);
  window.setTimeout(() => {
    statusText.textContent = 'Ready';
    addActivity('demo response: the ME3 agent would route this through Qwen and MCP tools');
  }, 350);
  input.value = '';
});

micButton.addEventListener('click', () => {
  addActivity('microphone placeholder clicked; connect the ME3 voice loop for live capture');
});

addActivity('system ready: dependency-free static demo loaded');
