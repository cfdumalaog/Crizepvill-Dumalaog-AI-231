const messageEl = document.getElementById("message");
const voiceEl = document.getElementById("voice");
const pitchEl = document.getElementById("pitch");
const rateEl = document.getElementById("rate");
const pitchVal = document.getElementById("pitchVal");
const rateVal = document.getElementById("rateVal");
const injectBtn = document.getElementById("injectBtn");
const meterFill = document.getElementById("meterFill");
const statusEl = document.getElementById("status");

let voices = [];
let progressTimer = null;

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
  const defIdx = voices.findIndex((v) => v.default);
  if (defIdx >= 0) voiceEl.value = String(defIdx);
}

loadVoices();
speechSynthesis.onvoiceschanged = loadVoices;

pitchEl.addEventListener("input", () => (pitchVal.textContent = pitchEl.value));
rateEl.addEventListener("input", () => (rateVal.textContent = rateEl.value));

function setMeter(pct) {
  meterFill.style.width = pct + "%";
}

function stopProgress() {
  clearInterval(progressTimer);
  progressTimer = null;
}

function startProgress() {
  stopProgress();
  setMeter(0);
  let pct = 0;
  progressTimer = setInterval(() => {
    pct = Math.min(100, pct + 2);
    setMeter(pct);
  }, 120);
}

injectBtn.addEventListener("click", () => {
  const text = messageEl.value.trim();
  if (!text) {
    statusEl.textContent = "Please type a message first!";
    return;
  }
  if (!("speechSynthesis" in window)) {
    statusEl.textContent = "Sorry, your browser does not support speech synthesis.";
    return;
  }

  speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  const idx = parseInt(voiceEl.value, 10);
  if (!Number.isNaN(idx) && voices[idx]) utter.voice = voices[idx];
  utter.pitch = parseFloat(pitchEl.value);
  utter.rate = parseFloat(rateEl.value);

  injectBtn.disabled = true;
  meterFill.classList.add("speaking");
  startProgress();
  statusEl.textContent = "Injecting voice… 🔊";

  utter.onend = utter.onerror = () => {
    injectBtn.disabled = false;
    meterFill.classList.remove("speaking");
    stopProgress();
    setMeter(100);
    statusEl.textContent = "Injection complete! ✅";
    setTimeout(() => setMeter(0), 1200);
  };

  speechSynthesis.speak(utter);
});
