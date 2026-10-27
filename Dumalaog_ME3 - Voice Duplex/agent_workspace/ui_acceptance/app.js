(function () {
  "use strict";

  const STORAGE_KEY = "ui_acceptance.todos";

  const form = document.getElementById("todo-form");
  const input = document.getElementById("todo-input");
  const list = document.getElementById("todo-list");
  const countEl = document.getElementById("todo-count");
  const clearBtn = document.getElementById("clear-completed");
  const filterButtons = Array.from(document.querySelectorAll(".filters__btn"));

  let todos = loadTodos();
  let filter = "all";

  function loadTodos() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      return [];
    }
  }

  function saveTodos() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(todos));
    } catch (err) {
      // Storage unavailable; app still works in memory.
    }
  }

  function uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }

  function visibleTodos() {
    if (filter === "active") return todos.filter((t) => !t.completed);
    if (filter === "completed") return todos.filter((t) => t.completed);
    return todos;
  }

  function render() {
    list.innerHTML = "";
    const items = visibleTodos();

    if (items.length === 0) {
      const li = document.createElement("li");
      li.className = "todo-empty";
      li.textContent =
        filter === "all"
          ? "No todos yet. Add one above!"
          : "Nothing here.";
      list.appendChild(li);
    }

    items.forEach((todo) => {
      const li = document.createElement("li");
      li.className = "todo-item" + (todo.completed ? " is-completed" : "");

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = todo.completed;
      checkbox.setAttribute("aria-label", "Mark " + todo.text + " complete");
      checkbox.addEventListener("change", () => toggleTodo(todo.id));

      const span = document.createElement("span");
      span.className = "todo-item__text";
      span.textContent = todo.text;

      const del = document.createElement("button");
      del.className = "todo-item__delete";
      del.type = "button";
      del.textContent = "\u2715";
      del.setAttribute("aria-label", "Delete " + todo.text);
      del.addEventListener("click", () => deleteTodo(todo.id));

      li.appendChild(checkbox);
      li.appendChild(span);
      li.appendChild(del);
      list.appendChild(li);
    });

    const remaining = todos.filter((t) => !t.completed).length;
    countEl.textContent = remaining + (remaining === 1 ? " item left" : " items left");
  }

  function addTodo(text) {
    todos.push({ id: uid(), text: text, completed: false });
    saveTodos();
    render();
  }

  function toggleTodo(id) {
    const todo = todos.find((t) => t.id === id);
    if (todo) {
      todo.completed = !todo.completed;
      saveTodos();
      render();
    }
  }

  function deleteTodo(id) {
    todos = todos.filter((t) => t.id !== id);
    saveTodos();
    render();
  }

  function clearCompleted() {
    todos = todos.filter((t) => !t.completed);
    saveTodos();
    render();
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    addTodo(text);
    input.value = "";
    input.focus();
  });

  filterButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      filter = btn.dataset.filter;
      filterButtons.forEach((b) => b.classList.toggle("is-active", b === btn));
      render();
    });
  });

  clearBtn.addEventListener("click", clearCompleted);

  render();
})();
