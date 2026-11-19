/* Conversation and contextual setup for the Pi-hosted ME3 Assistant.
 * Credentials go only to dedicated setup endpoints, never to /api/chat.
 */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const feed = el('chatFeed');
  const input = el('chatInput');
  const seen = new Set();
  let sessionId = null;
  const layerLabels = {
    asr: {
      qwen_remote: 'Qwen2-Audio DGX3', deepgram: 'Deepgram Nova-3',
      faster_whisper: 'faster-whisper local', openai: 'OpenAI transcription',
      custom_vllm: 'Custom vLLM/API'
    },
    tts: {
      qwen_remote: 'Qwen3-TTS Aiden', piper: 'Piper local',
      elevenlabs: 'ElevenLabs Filipino', openai: 'OpenAI Onyx',
      custom_vllm: 'Custom vLLM/API'
    },
    vad: {silero: 'Silero local', energy: 'Energy gate fallback'}
  };
  const flows = {
    openai: {
      title: 'OpenAI voice provider',
      help: 'Enter the API key here through the local ME3 tunnel. It goes directly to private Pi storage. Then launch OpenAI to use its ASR, tool model, and voice.',
      status: '/api/provider/status',
      fields: [{name:'api_key', label:'OpenAI API key', secret:true}],
      save: '/api/integrations/openai-key',
      actions: [{label:'Launch OpenAI', run:() => activateProvider('openai')}],
    },
    qwen: {
      title: 'Qwen voice provider',
      help: 'Qwen uses the institutional ASR, text, and TTS endpoints through the Pi VPN. Launch it when those servers are reachable.',
      status: '/api/provider/status',
      fields: [],
      actions: [{label:'Launch Qwen', run:() => activateProvider('qwen')}],
    },
    deepgram: {
      title: 'Deepgram speech recognition backup',
      help: 'Deepgram is an ASR backup for Qwen. It does not replace the text model or speech output.',
      status: '/api/deepgram/status',
      fields: [{name:'api_key', label:'Deepgram API key', secret:true}],
      save: '/api/integrations/deepgram-key',
    },
    spotify: {
      title: 'Spotify music',
      help: 'Save the Spotify app Client ID, then connect the account through OAuth. Playback requires a compatible Spotify player and account.',
      status: '/api/spotify/status',
      fields: [{name:'client_id', label:'Spotify app Client ID'}],
      save: '/api/spotify/client-id',
      actions: [{label:'Connect Spotify', run:() => location.assign('/api/spotify/connect')}],
    },
    google: {
      title: 'Google Calendar',
      help: 'Save your Google OAuth web client, then connect Calendar for reminders and meetings. Register the callback URL shown below in Google Cloud.',
      status: '/api/google/status',
      fields: [{name:'client_id', label:'OAuth Client ID'},
               {name:'client_secret', label:'OAuth Client Secret', secret:true}],
      save: '/api/integrations/google-config',
      actions: [{label:'Connect Calendar', run:() => location.assign('/api/google/connect')}],
    },
    telegram: {
      title: 'Telegram messages',
      help: 'Create a bot with BotFather, save its token, then have the recipient press Start in the bot. ME3 sends as the bot, not your personal Telegram account.',
      status: '/api/telegram/status',
      fields: [{name:'bot_token', label:'BotFather token', secret:true}],
      save: '/api/integrations/telegram-token',
      actions: [{label:'Connect @mncfrndz', run:() => postAction('/api/telegram/connect-target')}],
    },
    github: {
      title: 'GitHub publishing',
      help: 'Save a token with access to the repositories you want ME3 to use. ME3 can then verify access and publish requested sites.',
      status: '/api/github/status',
      fields: [{name:'token', label:'GitHub token', secret:true}],
      save: '/api/integrations/github-token',
    },
  };

  function message(role, text, meta='') {
    const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 90;
    const box = document.createElement('div');
    box.className = 'chat-message ' + role;
    if(meta) {
      const label = document.createElement('div');
      label.className = 'chat-meta'; label.textContent = meta; box.appendChild(label);
    }
    const body = document.createElement('div'); body.textContent = String(text || '');
    box.appendChild(body); feed.appendChild(box);
    if(atBottom) feed.scrollTop = feed.scrollHeight;
    return box;
  }

  function button(label, action, parent) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = label;
    b.addEventListener('click', action); parent.appendChild(b); return b;
  }

  function secretLike(text) {
    return /\b(?:sk-[A-Za-z0-9_-]{16,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|\d{5,16}:[A-Za-z0-9_-]{20,})\b/i.test(text)
      || /\b(?:api[ _-]?key|client[ _-]?secret|bot[ _-]?token|password)\s*(?::|=|\bis\b)\s*\S{8,}/i.test(text)
      || /^[a-f0-9]{32,}$/i.test(text.trim());
  }

  function setupIntent(text) {
    const t = String(text || '').toLowerCase();
    if(!/\b(set\s*up|setup|configure|connect|switch|change provider|use openai|use qwen|add (?:an? )?(?:api|token)|credentials?)\b/.test(t)) return '';
    if(/\b(?:vllm|openai-compatible|custom provider|another provider|different provider|base url|model endpoint|model server)\b/.test(t)) return 'custom_vllm';
    if(/openai|gpt/.test(t)) return 'openai';
    if(/qwen|dgx/.test(t)) return 'qwen';
    if(/deepgram/.test(t)) return 'deepgram';
    if(/spotify/.test(t)) return 'spotify';
    if(/google|calendar/.test(t)) return 'google';
    if(/telegram|botfather/.test(t)) return 'telegram';
    if(/github/.test(t)) return 'github';
    return 'menu';
  }

  async function jsonRequest(url, options={}) {
    const response = await fetch(url, {cache:'no-store', ...options});
    let data;
    try { data = await response.json(); } catch(_) { data = {}; }
    if(!response.ok || data.ok === false) {
      if(response.status === 403) throw new Error('Open ME3 through the local SSH tunnel at http://127.0.0.1:8100/ to save settings.');
      throw new Error(data.error || 'The Pi did not accept this request.');
    }
    return data;
  }

  async function postAction(url, body) {
    const data = await jsonRequest(url, {method:'POST', headers:{'Content-Type':'application/json'},
      ...(body === undefined ? {} : {body:JSON.stringify(body)})});
    message('system', 'Action completed.');
    await refreshSummary();
    return data;
  }

  async function activateProvider(provider) {
    await jsonRequest('/api/provider/select', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({provider})});
    message('system', 'Switching to ' + provider.toUpperCase() + '. ME3 will reconnect after the Pi service restarts.');
    await jsonRequest('/api/restart-service', {method:'POST'});
    setTimeout(refreshSummary, 5000);
  }

  function promptBase(title, help) {
    const box = document.createElement('div'); box.className = 'setup-prompt';
    const h = document.createElement('h3'); h.textContent = title; box.appendChild(h);
    const p = document.createElement('p'); p.textContent = help; box.appendChild(p);
    const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 90;
    feed.appendChild(box); if(atBottom) feed.scrollTop = feed.scrollHeight;
    return box;
  }

  function setupMenu() {
    const box = promptBase('What would you like to set up?',
      'Choose a service. ME3 will ask only for the fields needed for that service.');
    const choices = document.createElement('div'); choices.className = 'setup-choices'; box.appendChild(choices);
    for(const [id, spec] of Object.entries(flows)) button(spec.title, () => showSetup(id), choices);
    button('Other vLLM / API', () => showDynamicSetup('Use another vLLM or OpenAI-compatible provider.'), choices);
    const label = document.createElement('label'); label.textContent = 'Describe another provider or API';
    const description = document.createElement('textarea');
    description.rows = 2; description.maxLength = 1200; description.spellcheck = false;
    description.placeholder = 'Example: use this vLLM endpoint for the tool model and ask me only for the fields it needs.';
    label.appendChild(description); box.appendChild(label);
    button('Generate specific fields', () => {
      const request = description.value.trim();
      if(!request) { description.focus(); return; }
      showDynamicSetup(request);
    }, choices);
  }

  function makeDynamicField(field) {
    const label = document.createElement('label'); label.textContent = field.label || field.name;
    let entry;
    if(field.type === 'select') {
      entry = document.createElement('select');
      (field.options || []).forEach(option => {
        const item = document.createElement('option');
        item.value = option.value; item.textContent = option.label || option.value;
        entry.appendChild(item);
      });
    } else {
      entry = document.createElement('input');
      entry.type = field.type === 'password' ? 'password' : 'text';
      if(field.type === 'url') entry.inputMode = 'url';
    }
    entry.name = field.name; entry.autocomplete = field.autocomplete || (field.type === 'password' ? 'new-password' : 'off');
    entry.spellcheck = false; entry.placeholder = field.placeholder || '';
    if(field.default != null) entry.value = field.default;
    label.appendChild(entry); return {label, entry};
  }

  async function showDynamicSetup(request) {
    const safeRequest = String(request || '').trim();
    if(!safeRequest || secretLike(safeRequest)) {
      message('system', 'Describe the provider without pasting a key. The generated form has a private key field.');
      return setupMenu();
    }
    const box = promptBase('Generating a provider setup form…',
      'ME3 is converting your request into the smallest validated field set.');
    const status = document.createElement('div'); status.className = 'setup-status'; box.appendChild(status);
    try {
      const plan = await jsonRequest('/api/setup/plan', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({request:safeRequest})});
      box.replaceChildren();
      const h = document.createElement('h3'); h.textContent = plan.title || 'Custom provider setup'; box.appendChild(h);
      const p = document.createElement('p'); p.textContent = plan.help || 'Enter the requested provider details.'; box.appendChild(p);
      const values = {};
      (plan.fields || []).forEach(field => { const made = makeDynamicField(field); box.appendChild(made.label); values[field.name] = made.entry; });
      const actions = document.createElement('div'); actions.className = 'setup-actions'; box.appendChild(actions);
      const state = document.createElement('div'); state.className = 'setup-status'; box.appendChild(state);
      let profileId = '';
      async function saveProfile() {
        const payload = Object.fromEntries(Object.entries(values).map(([name, entry]) => [name, entry.value]));
        const result = await jsonRequest('/api/setup/profile', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
        profileId = result.activate_profile_id || '';
        Object.entries(values).forEach(([name, entry]) => { if(name === 'api_key') entry.value = ''; });
        state.textContent = 'Saved privately on the Pi. The key is not added to chat history.';
        return result;
      }
      button('Save privately on Pi', async ev => {
        ev.currentTarget.disabled = true;
        try { await saveProfile(); } catch(err) { state.textContent = err.message; }
        finally { ev.currentTarget.disabled = false; }
      }, actions);
      button('Save and activate', async ev => {
        ev.currentTarget.disabled = true;
        try {
          if(!profileId) await saveProfile();
          await jsonRequest('/api/setup/profile/activate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({profile_id:profileId})});
          state.textContent = 'Activated. ME3 is restarting with this provider; the current voice loop will reconnect shortly.';
          setTimeout(refreshSummary, 5000); setTimeout(loadLayers, 5000);
        } catch(err) { state.textContent = err.message; }
        finally { ev.currentTarget.disabled = false; }
      }, actions);
    } catch(err) { status.textContent = err.message; }
  }

  async function showSetup(id) {
    if(id === 'custom_vllm') return showDynamicSetup('Use another vLLM or OpenAI-compatible provider.');
    const spec = flows[id]; if(!spec) return setupMenu();
    const box = promptBase(spec.title, spec.help);
    const status = document.createElement('div'); status.className = 'setup-status';
    box.appendChild(status);
    const values = {};
    for(const field of spec.fields) {
      const label = document.createElement('label'); label.textContent = field.label;
      const entry = document.createElement('input'); entry.type = field.secret ? 'password' : 'text';
      entry.autocomplete = field.secret ? 'new-password' : 'off'; entry.spellcheck = false;
      label.appendChild(entry); box.appendChild(label); values[field.name] = entry;
    }
    const actions = document.createElement('div'); actions.className = 'setup-actions'; box.appendChild(actions);
    if(spec.save) button('Save privately on Pi', async ev => {
      const b = ev.currentTarget; b.disabled = true;
      const payload = Object.fromEntries(Object.entries(values).map(([name, entry]) => [name, entry.value]));
      try {
        await jsonRequest(spec.save, {method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify(payload)});
        status.textContent = 'Saved privately on the Pi. The value is not added to chat.';
        await refreshSummary();
      } catch(err) { status.textContent = err.message; }
      finally { Object.values(values).forEach(entry => {entry.value = '';}); b.disabled = false; }
    }, actions);
    for(const action of spec.actions || []) button(action.label, async ev => {
      const b = ev.currentTarget; b.disabled = true;
      try { await action.run(); status.textContent = 'Action started.'; }
      catch(err) { status.textContent = err.message; }
      finally { b.disabled = false; }
    }, actions);
    try {
      const data = await jsonRequest(spec.status);
      if(id === 'openai') status.textContent = 'Current: ' + data.selected + ' · OpenAI key ' + (data.openai_key_configured ? 'saved' : 'missing');
      else if(id === 'qwen') status.textContent = 'Current: ' + data.selected;
      else if(id === 'google') status.textContent = (data.authorized ? 'Calendar connected' : 'Calendar not connected') + (data.redirect_uri ? ' · Callback: ' + data.redirect_uri : '');
      else if(id === 'spotify') status.textContent = (data.connected ? 'Spotify connected' : 'Spotify not connected') + (data.redirect_uri ? ' · Callback: ' + data.redirect_uri : '');
      else status.textContent = data.configured ? 'Configured on Pi' : 'Setup needed';
    } catch(err) { status.textContent = err.message; }
  }

  async function sendChat(text) {
    const value = String(text || '').trim(); if(!value) return;
    if(secretLike(value)) {
      message('system', 'That looks like a credential. Choose its service below and enter it in the private setup prompt.');
      setupMenu(); return;
    }
    const intent = setupIntent(value);
    if(intent) {
      message('user', value, 'typed');
      if(intent === 'menu') setupMenu(); else if(intent === 'custom_vllm') showDynamicSetup(value); else showSetup(intent);
      return;
    }
    try {
      await jsonRequest('/api/chat', {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:value})});
    } catch(err) {
      message('system', err.message === 'use_private_setup_prompt'
        ? 'Use the private setup prompt for credentials.' : err.message);
      if(err.message === 'use_private_setup_prompt') setupMenu();
    }
  }

  function render(state) {
    if(state.session_id && sessionId && state.session_id !== sessionId) {
      seen.clear(); feed.replaceChildren();
      message('system', 'ME3 restarted. This is a new live conversation.');
    }
    if(state.session_id) sessionId = state.session_id;
    for(const event of state.conversation || []) {
      if(seen.has(event.id)) continue;
      seen.add(event.id);
      const role = event.role === 'user' ? 'user' : 'assistant';
      const meta = role === 'user' ? (event.source === 'chat' ? 'typed' : 'Pi microphone')
        : (event.kind === 'backchannel' ? 'progress' : 'ME3');
      const box = message(role, event.text, meta);
      if(role === 'user' && event.source !== 'chat') {
        const intent = setupIntent(event.text);
        if(intent) { if(intent === 'menu') setupMenu(); else if(intent === 'custom_vllm') showDynamicSetup(event.text); else showSetup(intent); }
      }
      if(role === 'assistant' && /\?/.test(event.text) &&
         /\b(confirm|approve|shall i|should i|would you like me to|go ahead|permission)\b/i.test(event.text)) {
        const choices = document.createElement('div'); choices.className = 'setup-choices'; box.appendChild(choices);
        button('Yes, proceed', () => sendChat('Yes, go ahead.'), choices);
        button('No, cancel', () => sendChat('No, cancel that.'), choices);
      }
      if(role === 'assistant' && /\b(not configured|missing (?:an? )?(?:api )?key|need (?:an? )?(?:api )?key)\b/i.test(event.text)) {
        const service = Object.keys(flows).find(name => event.text.toLowerCase().includes(name));
        if(service) showSetup(service);
      }
    }
    if(seen.size > 200) {
      const ids = [...seen]; ids.slice(0, ids.length - 150).forEach(id => seen.delete(id));
    }
  }

  async function refreshSummary() {
    try {
      const [provider, deepgram, spotify, google, telegram, github] = await Promise.all([
        '/api/provider/status','/api/deepgram/status','/api/spotify/status',
        '/api/google/status','/api/telegram/status','/api/github/status'
      ].map(url => jsonRequest(url).catch(() => ({}))));
      activeProvider = provider.selected || 'qwen';
      el('variantPill').textContent = activeProvider === 'openai' ? 'ME3 OPENAI' : 'ME3 QWEN INSTRUCT';
      const layers = await jsonRequest('/api/layers').catch(() => ({}));
      const selected = layers.layers || {};
      const layerText = selected.asr ? ' · Layers: ' + selected.asr + '/' + selected.tts + '/' + selected.vad : '';
      el('connectionSummary').textContent = 'Voice: ' + activeProvider.toUpperCase()
        + ' · OpenAI key: ' + (provider.openai_key_configured ? 'saved' : 'missing')
        + ' · Deepgram: ' + (deepgram.configured ? 'ready' : 'missing')
        + ' · Spotify: ' + (spotify.connected ? 'connected' : 'setup')
        + ' · Calendar: ' + (google.authorized ? 'connected' : 'setup')
        + ' · Telegram: ' + (telegram.configured ? 'connected' : 'setup')
        + ' · GitHub: ' + (github.configured ? 'ready' : 'setup') + layerText;
    } catch(_) { el('connectionSummary').textContent = 'Connection status unavailable.'; }
  }

  function populateLayerSelect(id, name, options, value) {
    const select = el(id); if(!select) return;
    select.replaceChildren();
    for(const option of options || []) {
      const item = document.createElement('option');
      item.value = option;
      item.textContent = (layerLabels[name] && layerLabels[name][option]) || option;
      select.appendChild(item);
    }
    if(value && options && options.includes(value)) select.value = value;
  }

  async function loadLayers() {
    try {
      const data = await jsonRequest('/api/layers');
      const values = data.layers || {}, options = data.options || {};
      populateLayerSelect('layerAsr', 'asr', options.asr, values.asr);
      populateLayerSelect('layerTts', 'tts', options.tts, values.tts);
      populateLayerSelect('layerVad', 'vad', options.vad, values.vad);
      el('layerStatus').textContent = 'Active: ' + (values.asr || '-') + ' · ' + (values.tts || '-') + ' · ' + (values.vad || '-');
    } catch(err) { el('layerStatus').textContent = 'Layer providers unavailable: ' + err.message; }
  }

  el('applyLayers').addEventListener('click', async () => {
    const button = el('applyLayers'), label = el('layerStatus');
    button.disabled = true; label.textContent = 'Saving layers privately on the Pi…';
    try {
      const result = await jsonRequest('/api/layers', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({asr:el('layerAsr').value, tts:el('layerTts').value, vad:el('layerVad').value})});
      label.textContent = result.restart_scheduled
        ? 'Saved. ME3 is restarting with the selected layers; reconnecting shortly.'
        : 'Saved. Restart the ME3 service to load the selected layers.';
      message('system', 'Model layers saved privately on the Pi. The service will reload them before the next voice turn.');
      setTimeout(loadLayers, 5000);
    } catch(err) { label.textContent = 'Layer update failed: ' + err.message; }
    finally { button.disabled = false; }
  });

  el('chatForm').addEventListener('submit', ev => {
    ev.preventDefault(); const value = input.value; input.value = ''; sendChat(value);
  });
  input.addEventListener('keydown', ev => {
    if(ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); el('chatForm').requestSubmit(); }
  });
  el('chatSetup').addEventListener('click', setupMenu);
  el('openSetup').addEventListener('click', setupMenu);
  window.ME3Chat = {render, showSetup, setupMenu};
  refreshSummary(); loadLayers(); setInterval(refreshSummary, 15000); setInterval(loadLayers, 15000);
})();
