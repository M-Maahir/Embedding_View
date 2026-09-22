import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

const CLUSTER_COLORS = [
  0xe2a33a,
  0x6cb3d9,
  0xd46b6b,
  0x7cbc8a,
  0xb48ad4,
  0xd4c36a,
];

const ANIM_MS = 2500;
const SCATTER_RADIUS = 7;

const wordInput = document.getElementById("word-input");
const inputLabel = document.getElementById("input-label");
const inputHint = document.getElementById("input-hint");
const modeWordsBtn = document.getElementById("mode-words");
const modeSentencesBtn = document.getElementById("mode-sentences");
const embedBtn = document.getElementById("embed-btn");
const replayBtn = document.getElementById("replay-btn");
const statusEl = document.getElementById("status");
const legendSection = document.getElementById("legend-section");
const legendEl = document.getElementById("legend");
const neighborSection = document.getElementById("neighbor-section");
const neighborTitle = document.getElementById("neighbor-title");
const neighborsEl = document.getElementById("neighbors");
const viewport = document.getElementById("viewport");

const WORD_SAMPLE = `king, queen, man, woman, prince, princess
cat, dog, kitten, puppy
apple, orange, banana, grape
car, truck, bus, train
happy, sad, joy, anger`;

const SENTENCE_SAMPLE = `The king sat beside the queen in the old palace.
A prince and a princess argued about the crown.
The cat chased the kitten across the kitchen.
A dog and a puppy waited by the door.
Apples, oranges, bananas, and grapes filled the market stalls.
A car and a truck sat behind the bus at the train crossing.
She felt happy at first, then sad, then a sudden rush of joy and anger.`;

const MODES = {
  words: {
    label: "Words",
    hint: "Comma or newline separated. Cap 80.",
    sample: WORD_SAMPLE,
    unit: "word",
  },
  sentences: {
    label: "Sentences",
    hint: "Paste a paragraph. NLTK splits it into sentences, max 80.",
    sample: SENTENCE_SAMPLE,
    unit: "sentence",
  },
};

const state = {
  mode: "words",
  payload: null,
  nodes: [],
  hoverLines: null,
  hovered: null,
  animStart: 0,
  animating: false,
  pulseUntil: 0,
  pulseCluster: null,
};

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
camera.position.set(9, 6.5, 11);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
viewport.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = "absolute";
labelRenderer.domElement.style.inset = "0";
labelRenderer.domElement.style.pointerEvents = "none";
viewport.appendChild(labelRenderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x9aa4b8, 0.85));
const keyLight = new THREE.DirectionalLight(0xfff4e0, 0.9);
keyLight.position.set(6, 10, 4);
scene.add(keyLight);

const grid = new THREE.GridHelper(16, 16, 0x3a4254, 0x232a38);
grid.position.y = -5;
scene.add(grid);

const wordGroup = new THREE.Group();
scene.add(wordGroup);

const sphereGeom = new THREE.SphereGeometry(0.16, 24, 16);
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function setStatus(text) {
  statusEl.textContent = text;
}

function parseWords(text) {
  const seen = new Set();
  const words = [];
  for (const part of text.split(/[\n,]+/)) {
    const word = part.trim();
    if (!word) continue;
    const key = word.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    words.push(word);
    if (words.length >= 80) break;
  }
  return words;
}

function setMode(mode) {
  if (!MODES[mode] || state.mode === mode) {
    if (MODES[mode]) {
      state.mode = mode;
    }
    return;
  }
  const previous = MODES[state.mode];
  const current = wordInput.value.replace(/\r\n/g, "\n").trim();
  if (current === previous.sample.replace(/\r\n/g, "\n").trim()) {
    wordInput.value = MODES[mode].sample;
  }
  state.mode = mode;
  inputLabel.textContent = MODES[mode].label;
  inputHint.textContent = MODES[mode].hint;
  modeWordsBtn.classList.toggle("is-active", mode === "words");
  modeSentencesBtn.classList.toggle("is-active", mode === "sentences");
  modeWordsBtn.setAttribute("aria-selected", String(mode === "words"));
  modeSentencesBtn.setAttribute("aria-selected", String(mode === "sentences"));
}

function easeOutCubic(t) {
  return 1 - (1 - t) ** 3;
}

function randomOnSphere(radius) {
  const u = Math.random();
  const v = Math.random();
  const theta = 2 * Math.PI * u;
  const phi = Math.acos(2 * v - 1);
  return new THREE.Vector3(
    radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi),
  );
}

function clusterColor(id) {
  return CLUSTER_COLORS[id % CLUSTER_COLORS.length];
}

function hexCss(hex) {
  return `#${hex.toString(16).padStart(6, "0")}`;
}

function clearNodes() {
  for (const node of state.nodes) {
    wordGroup.remove(node.mesh);
    node.mesh.geometry = sphereGeom;
    node.mesh.material.dispose();
    if (node.label.element.parentNode) {
      node.label.element.remove();
    }
  }
  state.nodes = [];
  clearHoverLines();
}

function clearHoverLines() {
  if (state.hoverLines) {
    wordGroup.remove(state.hoverLines);
    state.hoverLines.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
    state.hoverLines = null;
  }
}

function placeWords(payload, scatter) {
  clearNodes();
  state.payload = payload;

  for (const point of payload.points) {
    const material = new THREE.MeshStandardMaterial({
      color: clusterColor(point.cluster),
      roughness: 0.38,
      metalness: 0.12,
      emissive: clusterColor(point.cluster),
      emissiveIntensity: 0.12,
    });
    const mesh = new THREE.Mesh(sphereGeom, material);
    mesh.userData.word = point.word;
    mesh.userData.cluster = point.cluster;

    const target = new THREE.Vector3(point.x, point.y, point.z);
    const start = scatter ? randomOnSphere(SCATTER_RADIUS) : target.clone();
    mesh.position.copy(start);

    const labelEl = document.createElement("div");
    labelEl.className = "word-label";
    const display = point.label || point.word;
    labelEl.textContent = display;
    labelEl.title = point.word;
    const label = new CSS2DObject(labelEl);
    label.position.set(0, 0.28, 0);
    mesh.add(label);
    wordGroup.add(mesh);

    state.nodes.push({
      word: point.word,
      cluster: point.cluster,
      mesh,
      label,
      labelEl,
      start,
      target,
    });
  }

  renderLegend(payload.points);
  startAnimation();
  replayBtn.disabled = false;
}

function renderLegend(points) {
  const counts = new Map();
  for (const point of points) {
    counts.set(point.cluster, (counts.get(point.cluster) || 0) + 1);
  }
  legendEl.innerHTML = "";
  for (const [cluster, count] of [...counts.entries()].sort((a, b) => a[0] - b[0])) {
    const li = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = hexCss(clusterColor(cluster));
    li.append(swatch, document.createTextNode(`Group ${cluster + 1} · ${count}`));
    legendEl.appendChild(li);
  }
  legendSection.hidden = counts.size === 0;
}

function startAnimation() {
  for (const node of state.nodes) {
    node.start = node.mesh.position.clone();
  }
  state.animStart = performance.now();
  state.animating = true;
  setStatus("Animating toward embedding positions…");
}

function replayAnimation() {
  if (!state.payload) return;
  for (const node of state.nodes) {
    node.start = randomOnSphere(SCATTER_RADIUS);
    node.mesh.position.copy(node.start);
    node.mesh.scale.setScalar(1);
  }
  clearHover();
  startAnimation();
}

function animationProgress() {
  if (!state.animating) return 1;
  const t = Math.min(1, (performance.now() - state.animStart) / ANIM_MS);
  return easeOutCubic(t);
}

function applyAnimation() {
  if (!state.animating) return;
  const t = animationProgress();
  for (const node of state.nodes) {
    node.mesh.position.lerpVectors(node.start, node.target, t);
  }
  if (t >= 1) {
    state.animating = false;
    setStatus(
      `Settled (${state.payload?.count || state.nodes.length} ${state.payload?.mode === "sentences" ? "sentences" : "words"}). Hover a point for nearest neighbors.`,
    );
  }
}

function nodeByWord(word) {
  return state.nodes.find((node) => node.word === word);
}

function showNeighbors(word) {
  const list = state.payload?.neighbors?.[word] || [];
  neighborTitle.textContent = word;
  neighborTitle.title = word;
  neighborsEl.innerHTML = "";
  if (!list.length) {
    const li = document.createElement("li");
    li.textContent = "No neighbors";
    neighborsEl.appendChild(li);
  } else {
    for (const item of list) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = item.word;
      const score = document.createElement("span");
      score.textContent = item.score.toFixed(2);
      li.append(name, score);
      neighborsEl.appendChild(li);
    }
  }
  neighborSection.hidden = false;
}

function drawNeighborLines(node) {
  clearHoverLines();
  const list = state.payload?.neighbors?.[node.word] || [];
  if (!list.length) return;

  const group = new THREE.Group();
  for (const item of list) {
    const other = nodeByWord(item.word);
    if (!other) continue;
    const geometry = new THREE.BufferGeometry().setFromPoints([
      node.mesh.position,
      other.mesh.position,
    ]);
    const material = new THREE.LineBasicMaterial({
      color: 0xe2a33a,
      transparent: true,
      opacity: 0.55,
    });
    group.add(new THREE.Line(geometry, material));
  }
  state.hoverLines = group;
  wordGroup.add(group);
}

function setLabelState(node, hoveredWord, neighborWords) {
  node.labelEl.classList.toggle("is-hovered", node.word === hoveredWord);
  node.labelEl.classList.toggle("is-neighbor", neighborWords.has(node.word));
}

function applyHoverVisual(hovered) {
  const neighborWords = new Set(
    (state.payload?.neighbors?.[hovered?.word] || []).map((item) => item.word),
  );
  for (const node of state.nodes) {
    const related =
      !hovered ||
      node.word === hovered.word ||
      neighborWords.has(node.word) ||
      node.cluster === hovered.cluster;
    node.mesh.material.opacity = related ? 1 : 0.22;
    node.mesh.material.transparent = true;
    setLabelState(node, hovered?.word, neighborWords);
  }
}

function clearHover() {
  state.hovered = null;
  clearHoverLines();
  neighborSection.hidden = true;
  for (const node of state.nodes) {
    node.mesh.material.opacity = 1;
    node.mesh.material.transparent = false;
    node.labelEl.classList.remove("is-hovered", "is-neighbor");
  }
}

function onPointerMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(state.nodes.map((node) => node.mesh));
  if (!hits.length) {
    if (state.hovered) clearHover();
    return;
  }
  const word = hits[0].object.userData.word;
  if (state.hovered?.word === word) {
    if (state.hovered) drawNeighborLines(state.hovered);
    return;
  }
  const node = nodeByWord(word);
  state.hovered = node;
  showNeighbors(word);
  applyHoverVisual(node);
  drawNeighborLines(node);
}

function onPointerClick() {
  if (!state.hovered) return;
  state.pulseUntil = performance.now() + 900;
  state.pulseCluster = state.hovered.cluster;
}

function applyPulse() {
  const now = performance.now();
  const pulsing = now < state.pulseUntil;
  for (const node of state.nodes) {
    if (pulsing && node.cluster === state.pulseCluster) {
      const wave = 1 + 0.28 * Math.sin((state.pulseUntil - now) / 70);
      node.mesh.scale.setScalar(wave);
      node.mesh.material.emissiveIntensity = 0.45;
    } else {
      node.mesh.scale.setScalar(1);
      node.mesh.material.emissiveIntensity = 0.12;
    }
  }
}

function resize() {
  const { clientWidth, clientHeight } = viewport;
  camera.aspect = clientWidth / Math.max(clientHeight, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight);
  labelRenderer.setSize(clientWidth, clientHeight);
}

async function embedItems() {
  const mode = state.mode;
  const unit = MODES[mode].unit;
  let body;
  if (mode === "sentences") {
    const text = wordInput.value.trim();
    if (!text) {
      setStatus("Paste at least one sentence.");
      return;
    }
    body = { mode, text };
  } else {
    const words = parseWords(wordInput.value);
    if (!words.length) {
      setStatus("Enter at least one word.");
      return;
    }
    body = { mode, words };
  }
  embedBtn.disabled = true;
  replayBtn.disabled = true;
  setStatus(`Tokenizing and embedding ${unit}s. First run may download MiniLM…`);
  try {
    const response = await fetch("/api/embed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      const detail = err.detail;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg || item).join(" ")
        : detail || `Request failed (${response.status})`;
      throw new Error(message);
    }
    const payload = await response.json();
    placeWords(payload, true);
  } catch (error) {
    setStatus(error.message || "Embed failed.");
    replayBtn.disabled = !state.payload;
  } finally {
    embedBtn.disabled = false;
  }
}

function tick() {
  applyAnimation();
  applyPulse();
  if (state.hovered && !state.animating) {
    drawNeighborLines(state.hovered);
  }
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  requestAnimationFrame(tick);
}

embedBtn.addEventListener("click", embedItems);
replayBtn.addEventListener("click", replayAnimation);
modeWordsBtn.addEventListener("click", () => setMode("words"));
modeSentencesBtn.addEventListener("click", () => setMode("sentences"));
renderer.domElement.addEventListener("pointermove", onPointerMove);
renderer.domElement.addEventListener("click", onPointerClick);
window.addEventListener("resize", resize);

resize();
tick();
setStatus("Ready. First embed downloads MiniLM (~80 MB).");
