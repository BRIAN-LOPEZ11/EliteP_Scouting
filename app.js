// Reglas de validación (deben coincidir con las de main_api.py)
const REGLAS = {
  edad: { min: 18, max: 45 },            // inclusive: 18 <= edad <= 45
  estatura: { min: 150, max: 210 }       // exclusivo: 150 < estatura < 210
};

const $ = id => document.getElementById(id);

// Traducción de posiciones para mostrar en pantalla
const TRADUCCION_POSICIONES = {
  'Attack': 'Delantero',
  'Defender': 'Defensa',
  'Midfield': 'Centrocampista',
  'Goalkeeper': 'Portero'
};

async function cargarPosiciones() {
  try {
    const r = await fetch('/api/posiciones');
    const d = await r.json();
    if (!r.ok || !Array.isArray(d.posiciones)) {
      throw new Error(r.status === 503 ? 'API apagada (puerto 8000)' : `Error ${r.status}`);
    }
    $('posicion').innerHTML = d.posiciones.map(p =>
      `<option value="${p}">${TRADUCCION_POSICIONES[p] || p}</option>`
    ).join('');
  } catch (e) {
    console.error('No se pudieron cargar las posiciones:', e);
    $('posicion').innerHTML = `<option value="">No disponible: ${e.message}</option>`;
  }
}

function mostrarError(texto) {
  $('error').textContent = texto;
  $('error').style.display = texto ? 'block' : 'none';
}

// Devuelve el cuerpo a enviar, o null si hay errores de validación
function validarFormulario() {
  const errores = [];

  // Validar nombre del jugador
  const nombre = $('nombreJugador').value.trim();

  if (nombre === '') {
    errores.push('Ingresa el nombre del jugador.');
  }

  if (!$('posicion').value) {
    errores.push('Selecciona una posición.');
  }

  const textoEdad = $('edad').value.trim();
  const edad = Number(textoEdad);

  if (textoEdad === '' || !Number.isInteger(edad)) {
    errores.push('La edad debe ser un número entero.');
  } else if (edad < REGLAS.edad.min || edad > REGLAS.edad.max) {
    errores.push(`La edad debe estar entre ${REGLAS.edad.min} y ${REGLAS.edad.max} años.`);
  }

  const textoEstatura = $('estatura').value.trim();
  let estatura = null;

  if (textoEstatura !== '') {
    estatura = Number(textoEstatura);

    if (Number.isNaN(estatura)) {
      errores.push('La estatura debe ser un número.');
    } else if (estatura <= REGLAS.estatura.min || estatura >= REGLAS.estatura.max) {
      errores.push(`La estatura debe ser mayor a ${REGLAS.estatura.min} cm y menor a ${REGLAS.estatura.max} cm.`);
    }
  }

  if (errores.length > 0) {
    mostrarError(errores.join(' '));
    return null;
  }

  const cuerpo = {
    posicion: $('posicion').value,
    edad: edad
  };

  if (estatura !== null) {
    cuerpo.estatura_cm = estatura;
  }

  return cuerpo;
}

function mensajeError(d) {
  if (Array.isArray(d.detail)) {
    return d.detail.map(e => `${e.loc[e.loc.length - 1]}: ${e.msg}`).join(' | ');
  }
  return d.detail || 'Error desconocido';
}

$('btn').addEventListener('click', async () => {
  mostrarError('');
  $('resultado').style.display = 'none';

  const cuerpo = validarFormulario();
  if (cuerpo === null) return;

  // Obtener nombre para personalizar el resultado
  const nombre = $('nombreJugador').value.trim();

  $('btn').disabled = true;

  try {
    const r = await fetch('/api', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cuerpo)
    });

    const d = await r.json();

    if (!r.ok) throw new Error(mensajeError(d));

    const prob = d.probabilidad_elite;

    // Mostrar clasificación personalizada
    $('etiqueta').textContent = `${nombre} es ${d.etiqueta}`;

    // Diferenciar visualmente Élite y Estándar
    $('etiqueta').className = 'etiqueta' + (d.prediccion === 0 ? ' standard' : '');

    // Mostrar porcentaje
    $('probabilityPlaceholder').textContent = `${prob.toFixed(1)}%`;

    // Mostrar detalles de la predicción
    $('detalle').textContent =
      `Estatura usada: ${d.estatura_usada_cm} cm · Umbral Élite: €${d.umbral_elite_eur.toLocaleString('es-CO')}`;

    // Color de la barra según probabilidad
    $('progreso').style.background = prob >= 50 ? 'var(--green)' : '#e67e22';

    $('progreso').style.width = '0';

    // Ocultar estado inicial si existe
    if ($('emptyState')) {
      $('emptyState').style.display = 'none';
    }

    $('resultado').style.display = 'block';

    requestAnimationFrame(() => {
      $('progreso').style.width = prob + '%';
    });

  } catch (e) {
    mostrarError(e.message);
  } finally {
    $('btn').disabled = false;
  }
});

cargarPosiciones();
