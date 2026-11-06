const cats = [
  { emoji: "🐱", name: "Mochi", vibe: "Professional napper" },
  { emoji: "😺", name: "Sir Whiskers", vibe: "Dignified loaf" },
  { emoji: "😸", name: "Biscuit", vibe: "Chaos gremlin" },
  { emoji: "🐈", name: "Luna", vibe: "Window watcher" },
  { emoji: "🐈‍⬛", name: "Midnight", vibe: "Mysterious shadow" },
  { emoji: "😻", name: "Cupcake", vibe: "Certified sweetheart" },
  { emoji: "🙀", name: "Panic", vibe: "Startled by cucumber" },
  { emoji: "😼", name: "Bandit", vibe: "Knocks things off tables" },
  { emoji: "🐈‍⬛", name: "Inky", vibe: "Sits in dark corners" },
  { emoji: "😽", name: "Honey", vibe: "Purrs at 260 Hz" },
  { emoji: "🐱", name: "Tofu", vibe: "Softest landing ever" },
  { emoji: "😾", name: "Grumpy", vibe: "Judges your life choices" },
];

const facts = [
  "Cats sleep for about 70% of their lives. That's roughly 11 hours a day, every day.",
  "A group of cats is called a 'clowder', and a group of kittens is a 'kindle'.",
  "Cats can rotate their ears 180 degrees, thanks to 32 muscles in each ear.",
  "A cat's purr vibrates at 25–150 Hz, a frequency shown to help heal bones and tissue.",
  "Cats have been domesticated for about 10,000 years — longer than dogs in many regions.",
  "Your cat can't taste sweetness. That's why they ignore your cake and eat the wrapper.",
  "Cats can jump up to six times their body length in a single leap.",
  "The cat's nose print is unique, just like a human fingerprint.",
  "Cats have a special third eyelid called a nictitating membrane.",
  "Ancient Egyptians mummified their cats. Some tombs held over 300,000 mummified cats.",
];

const nameParts = {
  first: ["Sir", "Lady", "Captain", "Duchess", "Professor", "Baron", "Miss", "General", "Baby", "King"],
  second: ["Whiskers", "Noodle", "Pickle", "Tofu", "Biscuit", "Mochi", "Waffles", "Pancake", "Gizmo", "Dumpling", "Miso", "Boba"],
};

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// Build the gallery
const grid = document.getElementById("cat-grid");
cats.forEach((cat) => {
  const card = document.createElement("div");
  card.className = "cat-card";
  card.setAttribute("role", "button");
  card.setAttribute("tabindex", "0");
  card.innerHTML = `
    <span class="emoji">${cat.emoji}</span>
    <div class="name">${cat.name}</div>
    <div class="vibe">${cat.vibe}</div>
  `;
  const meow = () => {
    const audio = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(600, audio.currentTime);
    osc.frequency.exponentialRampToValueAtTime(900, audio.currentTime + 0.08);
    osc.frequency.exponentialRampToValueAtTime(500, audio.currentTime + 0.25);
    gain.gain.setValueAtTime(0.15, audio.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.3);
    osc.connect(gain).connect(audio.destination);
    osc.start();
    osc.stop(audio.currentTime + 0.3);
    card.style.transform = "scale(1.1)";
    setTimeout(() => (card.style.transform = ""), 150);
  };
  card.addEventListener("click", meow);
  card.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      meow();
    }
  });
  grid.appendChild(card);
});

// Name generator
document.getElementById("generate-btn").addEventListener("click", () => {
  const name = `${pick(nameParts.first)} ${pick(nameParts.second)}`;
  document.getElementById("generated-name").textContent = `How about "${name}"? 🐾`;
});

// Facts
const factText = document.getElementById("fact-text");
let lastFact = -1;
function showFact() {
  let i;
  do {
    i = Math.floor(Math.random() * facts.length);
  } while (i === lastFact && facts.length > 1);
  lastFact = i;
  factText.textContent = facts[i];
}
document.getElementById("fact-btn").addEventListener("click", showFact);
showFact();
