const textEl = document.getElementById("text");
const voiceEl = document.getElementById("voice");
const rateEl = document.getElementById("rate");
const pitchEl = document.getElementById("pitch");
const rateVal = document.getElementById("rateVal");
const pitchVal = document.getElementById("pitchVal");
const injectBtn = document.getElementById("inject");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const waveEl = document.getElementById("wave");

let voices = [];

function loadVoices() {
  voices = speechSynthesis.getVoices();
  voiceEl.innerHTML = "";
  if (voices.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "Default voice";
    voiceEl.appendChild(opt);
    return;
  }
  voices.forEach((v, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${v.name} (${v.lang})${v.default ? " — default" : ""}`;
    voiceEl.appendChild(opt);
  });
  const def = voices.findIndex((v) => v.default);
  if (def >= 0) voiceEl.value = String(def);
  setStatus(`Loaded ${voices.length} voice${voices.length === 1 ? "" : "s"}. Ready to inject.`);
}

function setStatus(msg, speaking = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("speaking", speaking);
  waveEl.classList.toggle("active", speaking);
}

function speak() {
  const text = textEl.value.trim();
  if (!text) {
    setStatus("Type something first — nothing to inject!");
    return;
  }
  if (!("speechSynthesis" in window)) {
    setStatus("Sorry, this browser doesn't support the Web Speech API.");
    return;
  }
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  if (voices.length) u.voice = voices[Number(voiceEl.value)] || null;
  u.rate = Number(rateEl.value);
  u.pitch = Number(pitchEl.value);
  u.onstart = () => setStatus("Injecting voice… 🎶", true);
  u.onend = () => setStatus("Injection complete. Nice and colorful, right?");
  u.onerror = (e) => setStatus(`Voice error: ${e.error}`);
  speechSynthesis.speak(u);
}

function stop() {
  speechSynthesis.cancel();
  setStatus("Stopped. Silence restored.");
}

injectBtn.addEventListener("click", speak);
stopBtn.addEventListener("click", stop);
rateEl.addEventListener("input", () => (rateVal.textContent = Number(rateEl.value).toFixed(1)));
pitchEl.addEventListener("input", () => (pitchVal.textContent = Number(pitchEl.value).toFixed(1)));

if ("speechSynthesis" in window) {
  loadVoices();
  speechSynthesis.onvoiceschanged = loadVoices;
} else {
  setStatus("This browser doesn't support the Web Speech API. Try Chrome or Edge.");
}
