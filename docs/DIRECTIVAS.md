# 📋 DIRECTIVAS DE DESARROLLO Y DOCUMENTACIÓN

## Obligaciones y Procedimientos Mandatorios para Claude Code

Este archivo establece las **reglas obligatorias** que Claude debe seguir en **todas las conversaciones futuras** para mantener la consistencia, calidad y continuidad del proyecto.

---

## 🔧 METODOLOGÍA DE DESARROLLO OBLIGATORIA

### 1. **Flujo de Trabajo Mandatorio: Desarrollo → Test → Análisis → Documentación**

**SECUENCIA OBLIGATORIA** para cada módulo:

```
1. 📝 PLANIFICACIÓN
   - Revisar ANALYSIS_RESULTS.md para contexto
   - Identificar siguiente módulo según plan de implementación
   - Definir objetivos específicos y criterios de éxito

2. 🔨 IMPLEMENTACIÓN  
   - Desarrollar código siguiendo arquitectura modular existente
   - Mantener consistencia con convenciones establecidas
   - Implementar logging y manejo de errores robusto

3. 🧪 TESTING OBLIGATORIO
   - Crear script de test específico (test_[módulo].py)
   - Usar dataset de 500 canciones para validación
   - Ejecutar test ANTES de proceder al siguiente módulo
   - Analizar resultados y identificar problemas

4. 📊 ANÁLISIS DE RESULTADOS
   - Interpretar salidas técnicamente y en términos simples
   - Identificar hallazgos, limitaciones y mejoras
   - Validar que cumple objetivos establecidos

5. 📚 DOCUMENTACIÓN DUAL
   - Actualizar ANALYSIS_RESULTS.md con resultados del test
   - Actualizar DOCS.md con explicaciones académicas
   - Mantener coherencia entre ambos documentos
```

### 2. **Reglas de Testing Obligatorias**

**NUNCA** proceder al siguiente módulo sin:
- ✅ Test exitoso (≥80% funcionalidades)
- ✅ Resultados analizados e interpretados
- ✅ Documentación actualizada

**SIEMPRE** usar:
- Datos del dataset de 500 canciones (`tracks_features_500.csv`)
- Scripts de test específicos para cada módulo
- Logging detallado durante las pruebas
- Validación de calidad y performance

---

## 📝 OBLIGACIONES DE DOCUMENTACIÓN

### 1. **Actualización Mandatoria de ANALYSIS_RESULTS.md**

**CUÁNDO ACTUALIZAR** (OBLIGATORIO):
- ✅ Después de cada test exitoso
- ✅ Al identificar nuevos hallazgos o patrones
- ✅ Al completar cada módulo
- ✅ Al encontrar limitaciones o problemas

**QUÉ INCLUIR** (MANDATORIO):
- Fecha y hora de actualización
- Resultados numéricos específicos del test
- Interpretaciones técnicas y simples
- Implicaciones para el sistema general
- Próximos pasos actualizados

**FORMATO REQUERIDO**:
```markdown
### [N]. [Nombre del Módulo] (`[carpeta]/`)
**Estado**: ✅ Implementado y Validado  
**Test ejecutado**: `test_[módulo].py`  
**Fecha**: [YYYY-MM-DD]

#### Funcionalidades Validadas:
- ✅ [Funcionalidad 1]
- ✅ [Funcionalidad 2]
- ⚠️ [Funcionalidad limitada]

#### Resultados del Test:
```
[Resultados numéricos específicos]
```

#### Hallazgos Clave:
[Interpretaciones y análisis]
```

### 2. **Actualización Mandatoria de DOCS.md**

**CUÁNDO ACTUALIZAR** (OBLIGATORIO):
- ✅ Al introducir nuevos conceptos teóricos
- ✅ Al implementar nuevos algoritmos
- ✅ Al usar nuevas técnicas o metodologías
- ✅ Al obtener resultados que requieren explicación académica

**QUÉ INCLUIR** (MANDATORIO):
- **Marco teórico** de nuevos conceptos
- **Formulaciones matemáticas** cuando corresponda
- **Referencias bibliográficas** para conceptos utilizados
- **Justificaciones técnicas** de decisiones de diseño
- **Interpretaciones académicas** de resultados

**ESTILO REQUERIDO**:
- Formato académico riguroso (nivel tesis de ingeniería)
- Explicaciones progresivas (de conceptos básicos a avanzados)
- Citas bibliográficas cuando sea posible
- Formulaciones matemáticas en LaTeX cuando corresponda

---

## 🎯 CRITERIOS DE CALIDAD OBLIGATORIOS

### 1. **Testing y Validación**

**CRITERIOS MÍNIMOS** (OBLIGATORIOS):
- ✅ **Funcionalidad**: ≥80% de funciones operativas
- ✅ **Performance**: Tiempo de ejecución reasonable (<2min para 500 canciones)
- ✅ **Calidad de datos**: Sin errores críticos
- ✅ **Logging**: Información suficiente para debugging

**CRITERIOS DE EXCELENCIA** (DESEABLES):
- 🎯 **Funcionalidad**: 100% de funciones operativas
- 🎯 **Robustez**: Manejo de casos edge exitoso
- 🎯 **Interpretabilidad**: Resultados claramente explicables

### 2. **Documentación**

**ESTÁNDARES MÍNIMOS** (OBLIGATORIOS):
- ✅ **Completitud**: Todos los resultados documentados
- ✅ **Claridad**: Interpretaciones técnicas y simples incluidas
- ✅ **Actualización**: Fechas y versiones actualizadas
- ✅ **Consistencia**: Coherencia entre documentos

---

## 🔄 PROCEDIMIENTOS ESPECÍFICOS

### 1. **Al Iniciar Conversación**

**SIEMPRE** hacer en este orden:
1. Leer ANALYSIS_RESULTS.md para contexto actual
2. Identificar último módulo completado
3. Verificar próximo paso según plan de implementación
4. Confirmar con usuario antes de proceder

### 2. **Durante Desarrollo**

**MANTENER**:
- Arquitectura modular existente
- Convenciones de naming establecidas
- Estructura de carpetas definida
- Configuraciones centralizadas

**EVITAR**:
- Cambios en módulos ya validados
- Inconsistencias de estilo
- Hardcoding de parámetros
- Omitir logging o error handling

### 3. **Al Completar Módulo**

**SECUENCIA OBLIGATORIA**:
1. Ejecutar test completo
2. Analizar resultados detalladamente  
3. Actualizar ANALYSIS_RESULTS.md
4. Actualizar DOCS.md según corresponda
5. Confirmar todo funcionando antes de continuar

### 4. **Al Encontrar Problemas**

**PROCEDIMIENTO MANDATORIO**:
1. Documentar el problema específicamente
2. Intentar solución conservando funcionalidad existente
3. Si no es posible, solicitar orientación al usuario
4. Actualizar documentación con limitaciones encontradas

---

## 📚 OBLIGACIONES DE CONSISTENCIA

### 1. **Arquitectura y Código**

**MANTENER** siempre:
- Estructura modular: `config/`, `data_loading/`, `statistical_analysis/`, etc.
- Imports consistentes y organizados
- Docstrings en inglés para funciones
- Logging configurado apropiadamente
- Tests con nombres `test_[módulo].py`

### 2. **Documentación**

**SINCRONIZAR** siempre:
- ANALYSIS_RESULTS.md con resultados reales
- DOCS.md con conceptos utilizados
- CLAUDE.md con referencias actualizadas
- Estados de módulos consistentes entre documentos

### 3. **Datos**

**USAR** consistentemente:
- Dataset `tracks_features_500.csv` para tests
- Configuración: separador `;`, decimal `,`
- Sample sizes apropiados por módulo
- Métricas de calidad estandarizadas

---

## ⚠️ ADVERTENCIAS CRÍTICAS

### ❌ **NUNCA HACER**:
- Proceder sin ejecutar tests
- Modificar módulos ya validados sin justificación
- Omitir actualización de documentación
- Cambiar arquitectura establecida arbitrariamente
- Ignorar errores o warnings en tests

### ✅ **SIEMPRE HACER**:
- Leer contexto antes de implementar
- Validar funcionamiento antes de continuar
- Documentar hallazgos y limitaciones
- Mantener coherencia con trabajo previo
- Solicitar confirmación ante decisiones importantes

---

## 🎯 OBJETIVOS DE ESTAS DIRECTIVAS

1. **Continuidad**: Garantizar progreso coherente entre conversaciones
2. **Calidad**: Mantener estándares altos de código y documentación
3. **Trazabilidad**: Permitir seguimiento completo del desarrollo
4. **Reproducibilidad**: Facilitar replicación y validación
5. **Profesionalismo**: Mantener nivel académico/profesional

---

**ESTAS DIRECTIVAS SON OBLIGATORIAS Y DEBEN SEGUIRSE EN TODAS LAS CONVERSACIONES FUTURAS CON CLAUDE CODE**

---

*Establecidas: 26 de enero de 2025*  
*Versión: 1.0*  
*Próxima revisión: Al completar sistema de reportes*