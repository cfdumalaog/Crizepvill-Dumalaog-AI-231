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
    brain: {
      qwen: 'Qwen text/tool brain', openai: 'OpenAI text/tool brain',
      gemini: 'Gemini text/tool brain',
      custom_vllm: 'Configured vLLM/API profile'
    },
    asr: {
      qwen_remote: 'Qwen2-Audio DGX3', deepgram: 'Deepgram Nova-3',
      faster_whisper: 'faster-whisper local', openai: 'OpenAI transcription',
      custom_vllm: 'Custom vLLM/API'
    },
    tts: {
      qwen_remote: 'Qwen3-TTS Aiden', piper: 'Piper local',
      openai: 'OpenAI Onyx',
      custom_vllm: 'Custom vLLM/API'
    },
    vad: {silero: 'Silero local', energy: 'Energy gate fallback'}
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
    return /\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|\d{5,16}:[A-Za-z0-9_-]{20,})\b/i.test(text)
      || /\b(?:api[ _-]?key|client[ _-]?secret|bot[ _-]?token|password)\s*(?::|=|\bis\b)\s*\S{8,}/i.test(text)
      || /^[a-f0-9]{32,}$/i.test(text.trim());
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
    // Set up is a real conversation turn. The model decides what needs to be
    // configured and calls setup_form only after the target is clear.
    return sendChat(
      'I want to set something up. Ask me what provider, integration, or capability I want to configure. Do not show a fixed service menu; ask only the next useful question and generate a setup form after the target is clear.'
    ).then(ok => { if(!ok) renderProviderDownSetupPrompt(); return ok; });
  }

  function renderProviderDownSetupPrompt() {
    const box = promptBase(
      'What would you like to set up?',
      'The selected text/tool provider is unavailable right now. Describe the provider, integration, or capability in your own words; ME3 will generate only the fields that request needs.'
    );
    const entry = document.createElement('input');
    entry.type = 'text'; entry.autocomplete = 'off'; entry.spellcheck = false;
    entry.placeholder = 'Example: replace the Qwen vLLM key and use Qwen as the tool brain';
    box.appendChild(entry);
    const actions = document.createElement('div'); actions.className = 'setup-actions'; box.appendChild(actions);
    const state = document.createElement('div'); state.className = 'setup-status'; box.appendChild(state);
    const submit = button('Generate setup form', async () => {
      const request = entry.value.trim();
      if(!request) { state.textContent = 'Describe what you want to configure first.'; return; }
      submit.disabled = true; entry.disabled = true; state.textContent = 'Generating fields from your request…';
      try {
        const result = await jsonRequest('/api/setup/form', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body:JSON.stringify({request})
        });
        if(result.setup_form) {
          const form = {...result.setup_form, id: result.setup_form.id || ('offline-setup-' + Date.now())};
          message('assistant', result.message || 'I generated the fields needed for that setup request.', 'ME3');
          renderSetupForm(form);
          state.textContent = 'The generated form is ready below.';
        } else {
          state.textContent = result.question || result.error || 'Please describe one provider or integration.';
          entry.disabled = false; submit.disabled = false;
        }
      } catch(err) {
        state.textContent = err.message || 'The setup form could not be generated.';
        entry.disabled = false; submit.disabled = false;
      }
    }, actions);
    entry.addEventListener('keydown', ev => {
      if(ev.key === 'Enter') { ev.preventDefault(); submit.click(); }
    });
    entry.focus();
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

  const setupFormSeen = new Set();
  const secretFieldNames = new Set(['api_key', 'client_secret', 'bot_token', 'token', 'password']);

  function setupSummary(form, values) {
    const details = Object.entries(values)
      .filter(([name, value]) => value && !secretFieldNames.has(name))
      .map(([name, value]) => name + '=' + value).join(', ');
    return details || 'no non-secret values were supplied';
  }

  async function submitGeneratedSetup(form, values, state, buttonEl) {
    if(form._submitted) return;
    const fields = form.fields || [];
    for(const field of fields) {
      if(field.required !== false && !String(values[field.name] || '').trim()) {
        state.textContent = 'Please provide: ' + (field.label || field.name) + '.';
        return;
      }
    }
    buttonEl.disabled = true;
    try {
      const privateSave = form.private_save || null;
      let saved = null;
      let targetError = '';
      if(privateSave && privateSave.endpoint) {
        const privatePayload = {};
        (privateSave.fields || fields.map(field => field.name)).forEach(name => {
          if(values[name] !== undefined) privatePayload[name] = values[name];
        });
        saved = await jsonRequest(privateSave.endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(privatePayload)});
      }
      if(form.form_kind === 'telegram' && String(values.target || '').trim()) {
        try {
          await jsonRequest('/api/telegram/connect-target', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({username:String(values.target).trim()})});
        } catch(err) {
          targetError = err.message || 'Telegram target mapping needs one message to the bot.';
        }
      }
      if(form.activation && form.activation.provider) {
        await jsonRequest('/api/provider/select', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(form.activation)});
        await jsonRequest('/api/restart-service', {method:'POST'});
      } else if(saved && saved.activate_profile_id) {
        await jsonRequest('/api/setup/profile/activate', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({profile_id:saved.activate_profile_id})});
        await jsonRequest('/api/restart-service', {method:'POST'});
      }
      // Prevent double submission while the continuation turn is processed.
      form._submitted = true;
      buttonEl.textContent = 'Submitted';
      buttonEl.disabled = true;
      Object.keys(values).forEach(name => { if(secretFieldNames.has(name)) values[name] = ''; });
      state.textContent = targetError
        ? 'Token saved. Telegram could not map the target yet: ' + targetError
        : 'Submitted. Private fields were saved on the Pi; the non-secret details are continuing in the conversation.';
      await sendChat('I submitted the generated setup form for "' + (form.title || 'this setup') + '". Non-secret details: ' + setupSummary(form, values) + (targetError ? '. Telegram target mapping still needs attention: ' + targetError : '') + '. Any credential fields were entered through the protected setup card. This is a continuation of the same setup task; do not generate another setup form. Verify the saved configuration and tell me exactly what remains or whether it is ready.', {setupFollowup:true});
      await refreshSummary();
      setTimeout(loadLayers, 5000);
    } catch(err) {
      state.textContent = err.message;
    } finally {
      if(!form._submitted) buttonEl.disabled = false;
    }
  }

  function renderSetupForm(form) {
    if(!form || !form.id || setupFormSeen.has(form.id)) return;
    setupFormSeen.add(form.id);
    const box = promptBase(form.title || 'Setup details', form.help || 'ME3 generated the requirements for this setup request.');
    const note = document.createElement('div'); note.className = 'setup-status';
    note.textContent = 'Generated by the ToolBrain from this conversation. Non-secret values return to chat; secrets go only to protected Pi storage.';
    box.appendChild(note);
    const values = {};
    (form.fields || []).forEach(field => {
      const made = makeDynamicField(field); box.appendChild(made.label); values[field.name] = made.entry;
    });
    const actions = document.createElement('div'); actions.className = 'setup-actions'; box.appendChild(actions);
    const state = document.createElement('div'); state.className = 'setup-status'; box.appendChild(state);
    if((form.fields || []).length === 0 && !(form.activation || {}).provider) {
      state.textContent = 'No manual fields are required. Ask ME3 in the conversation for the next setup step.';
    }
    if((form.fields || []).length || (form.activation || {}).provider) {
      button('Submit in conversation', () => {
        const raw = Object.fromEntries(Object.entries(values).map(([name, entry]) => [name, entry.value]));
        submitGeneratedSetup(form, raw, state, actions.querySelector('button'));
      }, actions);
    }
  }

  async function sendChat(text, options={}) {
    const value = String(text || '').trim(); if(!value) return;
    if(secretLike(value)) {
      message('system', 'That looks like a credential. Ask ME3 to set up the service, then enter it in the generated private setup card.');
      return false;
    }
    try {
      await jsonRequest('/api/chat', {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:value, ...options})});
      return true;
    } catch(err) {
      message('system', err.message === 'use_private_setup_prompt'
        ? 'Use the generated private setup card for credentials.' : err.message);
      return false;
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
      const meta = role === 'user' ? (event.source === 'chat' ? 'typed' : 'voice microphone')
        : (event.kind === 'backchannel' ? 'progress' : 'ME3');
      const box = message(role, event.text, meta);
      if(role === 'assistant' && /\bapproval is required\b/i.test(event.text)) {
        const choices = document.createElement('div'); choices.className = 'setup-choices'; box.appendChild(choices);
        button('Approve action', () => sendChat('Approve.'), choices);
        button('Cancel action', () => sendChat('Cancel.'), choices);
      } else if(role === 'assistant' && /\?/.test(event.text) &&
         /\b(confirm|approve|shall i|should i|would you like me to|go ahead|permission)\b/i.test(event.text)) {
        const choices = document.createElement('div'); choices.className = 'setup-choices'; box.appendChild(choices);
        button('Yes, proceed', () => sendChat('Yes, go ahead.'), choices);
        button('No, cancel', () => sendChat('No, cancel that.'), choices);
      }
    }
    if(state.setup_form) renderSetupForm(state.setup_form);
    if(seen.size > 200) {
      const ids = [...seen]; ids.slice(0, ids.length - 150).forEach(id => seen.delete(id));
    }
  }

  async function refreshSummary() {
    try {
      const [provider, deepgram, spotify, google, telegram, github, permission] = await Promise.all([
        '/api/provider/status','/api/deepgram/status','/api/spotify/status',
        '/api/google/status','/api/telegram/status','/api/github/status','/api/permissions'
      ].map(url => jsonRequest(url).catch(() => ({}))));
      activeProvider = provider.selected || 'qwen';
      el('variantPill').textContent = activeProvider === 'openai' ? 'ME3 OPENAI' : 'ME3 QWEN INSTRUCT';
      const layers = await jsonRequest('/api/layers').catch(() => ({}));
      const selected = layers.layers || {};
      const layerText = selected.asr ? ' · Layers: ' + (selected.brain || '-') + '/' + selected.asr + '/' + selected.tts + '/' + selected.vad : '';
      el('connectionSummary').textContent = 'Voice: ' + activeProvider.toUpperCase()
        + ' · OpenAI key: ' + (provider.openai_key_configured ? 'saved' : 'missing')
        + ' · Deepgram: ' + (deepgram.configured ? 'ready' : 'missing')
        + ' · Spotify: ' + (spotify.connected ? 'connected' : 'setup')
        + ' · Calendar: ' + (google.authorized ? 'connected' : 'setup')
        + ' · Telegram: ' + (telegram.configured ? 'connected' : 'setup')
        + ' · GitHub: ' + (github.configured ? 'ready' : 'setup')
        + ' · Permissions: ' + (permission.label || permission.mode || 'unavailable') + layerText;
    } catch(_) { el('connectionSummary').textContent = 'Connection status unavailable.'; }
  }

  function populateLayerSelect(id, name, options, value, optionLabels={}) {
    const select = el(id); if(!select) return;
    select.replaceChildren();
    for(const option of options || []) {
      const item = document.createElement('option');
      const optionValue = typeof option === 'string' ? option : option.value;
      item.value = optionValue;
      item.textContent = optionLabels[optionValue]
        || (layerLabels[name] && layerLabels[name][optionValue]) || optionValue;
      select.appendChild(item);
    }
    if(value && [...select.options].some(option => option.value === value)) select.value = value;
  }

  async function loadLayers() {
    try {
      const data = await jsonRequest('/api/layers');
      const values = data.layers || {}, options = data.options || {}, labels = data.option_labels || {};
      populateLayerSelect('layerBrain', 'brain', options.brain, values.brain, labels);
      populateLayerSelect('layerAsr', 'asr', options.asr, values.asr, labels);
      populateLayerSelect('layerTts', 'tts', options.tts, values.tts, labels);
      populateLayerSelect('layerVad', 'vad', options.vad, values.vad, labels);
      el('layerStatus').textContent = 'Active: ' + (values.brain || '-') + ' · ' + (values.asr || '-') + ' · ' + (values.tts || '-') + ' · ' + (values.vad || '-');
    } catch(err) { el('layerStatus').textContent = 'Layer providers unavailable: ' + err.message; }
  }

  async function loadPermissions() {
    try {
      const data = await jsonRequest('/api/permissions');
      const select = el('permissionMode');
      if(data.mode && [...select.options].some(option => option.value === data.mode)) select.value = data.mode;
      const state = data.persisted ? 'saved privately on the Pi' : 'using the ME3 default';
      el('permissionStatus').innerHTML = '<strong>' + (data.label || data.mode || 'Unavailable')
        + '</strong> · ' + (data.description || '') + ' (' + state + ').';
    } catch(err) { el('permissionStatus').textContent = 'Permission policy unavailable: ' + err.message; }
  }

  el('applyLayers').addEventListener('click', async () => {
    const button = el('applyLayers'), label = el('layerStatus');
    button.disabled = true; label.textContent = 'Saving layers privately on the Pi…';
    try {
      const result = await jsonRequest('/api/layers', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({brain:el('layerBrain').value, asr:el('layerAsr').value, tts:el('layerTts').value, vad:el('layerVad').value})});
      label.textContent = result.restart_scheduled
        ? 'Saved. ME3 is restarting with the selected layers; reconnecting shortly.'
        : 'Saved. Restart the ME3 service to load the selected layers.';
      message('system', 'Model layers saved privately on the Pi. The service will reload them before the next voice turn.');
      setTimeout(loadLayers, 5000);
    } catch(err) { label.textContent = 'Layer update failed: ' + err.message; }
    finally { button.disabled = false; }
  });

  el('applyPermission').addEventListener('click', async () => {
    const button = el('applyPermission'), label = el('permissionStatus');
    button.disabled = true; label.textContent = 'Saving permission policy privately on the Pi…';
    try {
      const result = await jsonRequest('/api/permissions', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({mode:el('permissionMode').value})});
      label.innerHTML = '<strong>' + (result.label || result.mode) + '</strong> · '
        + (result.description || '') + ' (saved privately on the Pi).';
      message('system', 'ME3 permission mode saved: ' + (result.label || result.mode) + '.');
      refreshSummary();
    } catch(err) { label.textContent = 'Permission update failed: ' + err.message; }
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
  window.ME3Chat = {render, setupMenu, renderSetupForm};
  refreshSummary(); loadLayers(); loadPermissions();
  setInterval(refreshSummary, 15000); setInterval(loadLayers, 15000); setInterval(loadPermissions, 15000);
})();
