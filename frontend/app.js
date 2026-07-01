const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const recsEl = document.querySelector("#recommendations");
const recCountEl = document.querySelector("#recCount");
const turnCountEl = document.querySelector("#turnCount");
const clearButton = document.querySelector("#clearButton");
const healthStatus = document.querySelector("#healthStatus");

let conversation = [];
let lastRecommendations = [];

const typeLabels = {
  A: "Ability & Aptitude",
  B: "Biodata & Situational Judgement",
  C: "Competencies",
  D: "Development & 360",
  E: "Assessment Exercises",
  K: "Knowledge & Skills",
  P: "Personality & Behavior",
  S: "Simulations",
};

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function appendMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = role === "user" ? "U" : "A";
  article.innerHTML = `
    <div class="avatar">${label}</div>
    <div class="bubble">${escapeHtml(content)}</div>
  `;
  messagesEl.appendChild(article);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

let loadingBubbleEl = null;

function appendLoadingMessage() {
  if (loadingBubbleEl) return;
  const article = document.createElement("article");
  article.className = "message assistant loading-msg";
  article.innerHTML = `
    <div class="avatar" aria-hidden="true">A</div>
    <div class="bubble">
      <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>
  `;
  messagesEl.appendChild(article);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  loadingBubbleEl = article;
}

function removeLoadingMessage() {
  if (loadingBubbleEl) {
    loadingBubbleEl.remove();
    loadingBubbleEl = null;
  }
}

function renderRecommendations(items) {
  lastRecommendations = items || [];
  recCountEl.textContent = String(lastRecommendations.length);
  turnCountEl.textContent = String(conversation.length);

  if (!lastRecommendations.length) {
    recsEl.innerHTML = `
      <div class="empty-state">
        Recommendations appear here once the agent has enough role context.
      </div>
    `;
    return;
  }

  recsEl.innerHTML = lastRecommendations
    .map((rec) => {
      const typeLabel = typeLabels[rec.test_type] || rec.test_type;
      return `
        <article class="rec-card">
          <div class="rec-header">
            <h3 class="rec-title">${escapeHtml(rec.name)}</h3>
            <span class="type-badge" title="${escapeHtml(typeLabel)}">${escapeHtml(rec.test_type)}</span>
          </div>
          <a class="rec-link" href="${escapeHtml(rec.url)}" target="_blank" rel="noreferrer">Open catalog page</a>
        </article>
      `;
    })
    .join("");
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  input.disabled = isBusy;
  sendButton.querySelector("span").textContent = isBusy ? "Wait" : "Send";
}

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
}

async function sendMessage(content) {
  const clean = content.trim();
  if (!clean) return;

  conversation.push({ role: "user", content: clean });
  appendMessage("user", clean);
  renderRecommendations(lastRecommendations);
  input.value = "";
  autosize();
  setBusy(true);
  appendLoadingMessage();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: conversation }),
    });
    removeLoadingMessage();
    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }
    const data = await response.json();
    conversation.push({ role: "assistant", content: data.reply });
    appendMessage("assistant", data.reply);
    renderRecommendations(data.recommendations || []);
  } catch (error) {
    removeLoadingMessage();
    const text = "The backend did not return a valid response. Check the server log and try again.";
    appendMessage("assistant", text);
    recsEl.innerHTML = `<div class="empty-state error-text">${escapeHtml(error.message)}</div>`;
  } finally {
    setBusy(false);
    input.focus();
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    if (response.ok && data.status === "ok") {
      healthStatus.className = "status ok";
      healthStatus.lastElementChild.textContent = "Online";
      return;
    }
    throw new Error("Unhealthy");
  } catch {
    healthStatus.className = "status error";
    healthStatus.lastElementChild.textContent = "Offline";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("input", autosize);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

clearButton.addEventListener("click", () => {
  conversation = [];
  lastRecommendations = [];
  messagesEl.innerHTML = `
    <article class="message assistant">
      <div class="avatar">A</div>
      <div class="bubble">
        Tell me the role, skills, seniority, or constraints. I will return SHL catalog recommendations only.
      </div>
    </article>
  `;
  renderRecommendations([]);
  input.focus();
});

renderRecommendations([]);
checkHealth();
setInterval(checkHealth, 30000);

