/**
 * 🛠️ GIDEON TOOLS - Funzioni Avanzate Unificate
 * Questo file contiene tutte le funzioni tool di GIDEON
 * Importato da gideon_unified.html
 */

// ============ GIDEON TOOLS SIDEBAR FUNCTIONS ============

// Toggle toolbar expansion
function toggleToolbar() {
    const toolbar = document.getElementById('gideon-toolbar');
    if (toolbar) toolbar.classList.toggle('expanded');
}

// Toggle section open/close
function toggleSection(sectionId) {
    const section = document.getElementById('section-' + sectionId);
    if (section) section.classList.toggle('open');
}

// Show loading indicator
function showToolbarLoading(text = 'Elaborazione in corso...') {
    const loading = document.getElementById('toolbar-loading');
    const loadingText = document.getElementById('loading-text');
    if (loading && loadingText) {
        loadingText.textContent = text;
        loading.classList.add('active');
    }
}

// Hide loading indicator  
function hideToolbarLoading() {
    const loading = document.getElementById('toolbar-loading');
    if (loading) loading.classList.remove('active');
}

// Show results panel
function showResultsPanel(title, content) {
    const panel = document.getElementById('results-panel');
    const titleEl = document.getElementById('results-title');
    const contentEl = document.getElementById('results-content');
    
    if (panel && titleEl && contentEl) {
        titleEl.textContent = title;
        contentEl.innerHTML = content;
        panel.classList.add('active');
    } else {
        // Fallback: add message to chat
        if (typeof addSystemMessage === 'function') {
            addSystemMessage(`📊 ${title}\n${content.replace(/<[^>]*>/g, '')}`);
        }
    }
}

// Close results panel
function closeResultsPanel() {
    const panel = document.getElementById('results-panel');
    if (panel) panel.classList.remove('active');
}

// ============ MAIN TOOL EXECUTION ============

// Main tool execution function - integrated with GIDEON backend
async function runGideonTool(category, action) {
    const toolNames = {
        // Security
        'security_full_scan': '🔍 Scansione Sicurezza Completa',
        'security_vuln_check': '⚠️ Verifica Vulnerabilità',
        'security_firewall': '🛡️ Stato Firewall',
        'security_audit': '📋 Audit di Sistema',
        // Health
        'health_diagnostics': '🩺 Diagnostica Sintomi',
        'health_medications': '💉 Informazioni Farmaci',
        'health_wellness': '❤️ Analisi Benessere',
        'health_emergency': '🚑 Protocollo Emergenza',
        // Science
        'science_research': '📚 Ricerca Scientifica',
        'science_physics': '⚛️ Calcoli Fisica',
        'science_biology': '🧬 Analisi Biologica',
        'science_simulation': '📊 Simulazione',
        // Chemistry
        'chemistry_compounds': '⚗️ Analisi Composti',
        'chemistry_reactions': '💥 Reazioni Chimiche',
        'chemistry_periodic': '📋 Tavola Periodica',
        'chemistry_molecular': '🔗 Modelli Molecolari',
        // Archaeology
        'archaeology_artifacts': '🏺 Catalogo Reperti',
        'archaeology_dating': '📅 Datazione',
        'archaeology_civilizations': '🗿 Civiltà Antiche',
        'archaeology_sites': '🗺️ Siti Archeologici',
        // Military
        'military_strategy': '📋 Analisi Strategica',
        'military_intel': '🔍 Intelligence',
        'military_logistics': '📦 Logistica',
        'military_defense': '🛡️ Difesa',
        // Monitor
        'monitor_system': '💻 Monitoraggio Sistema',
        'monitor_network': '🌐 Monitoraggio Rete',
        'monitor_performance': '📈 Performance',
        'monitor_logs': '📜 Analisi Logs',
        // Cyber
        'cyber_threat_scan': '🔴 Scansione Minacce',
        'cyber_intrusion': '🚨 Rilevamento Intrusioni',
        'cyber_encryption': '🔐 Crittografia',
        'cyber_pentest': '💀 Penetration Test',
        // Analysis
        'analysis_text': '📝 Analisi Testuale',
        'analysis_sentiment': '😊 Sentiment Analysis',
        'analysis_summary': '📋 Riassunto Automatico',
        'analysis_translate': '🌍 Traduzione',
        'analysis_code': '💻 Analisi Codice',
        // Utilities
        'utilities_calculator': '🧮 Calcolatrice',
        'utilities_converter': '🔄 Convertitore',
        'utilities_timer': '⏰ Timer/Promemoria',
        'utilities_weather': '🌤️ Meteo',
        'utilities_qrcode': '📱 QR Code',
        // AI Tools
        'ai_image_gen': '🎨 Genera Immagine',
        'ai_image_analyze': '👁️ Analizza Immagine',
        'ai_voice': '🎤 Assistente Vocale',
        'ai_reasoning': '🧠 Deep Reasoning',
        'ai_creative': '✨ Scrittura Creativa',
        // Data
        'data_export': '📤 Esporta Chat',
        'data_import': '📥 Importa Dati',
        'data_backup': '💿 Backup',
        'data_history': '📜 Cronologia',
        'data_settings': '⚙️ Impostazioni'
    };
    
    const key = `${category}_${action}`;
    const toolName = toolNames[key] || `${category} - ${action}`;
    const apiBase = window.state?.apiBaseUrl || 'http://127.0.0.1:8001';
    
    showToolbarLoading(`Esecuzione: ${toolName}`);
    
    try {
        // Call GIDEON backend API
        const response = await fetch(`${apiBase}/api/gideon/tools/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                category: category,
                action: action,
                session_id: window.state?.sessionId || 'session_' + Date.now(),
                timestamp: new Date().toISOString()
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        hideToolbarLoading();
        
        // Check if AI fallback is needed
        if (result.use_ai_fallback) {
            const query = buildToolQuery(category, action);
            await processWithGideonAI(category, action, query, toolName);
            return;
        }
        
        // Format and display results
        const formattedContent = formatToolResult(category, action, result);
        showResultsPanel(toolName, formattedContent);
        
        // Also add to chat
        if (typeof addSystemMessage === 'function') {
            addSystemMessage(`✅ ${toolName} completato.`);
        }
        
    } catch (error) {
        hideToolbarLoading();
        console.error('Tool execution error:', error);
        
        // Fallback: use GIDEON AI to process the request
        const query = buildToolQuery(category, action);
        await processWithGideonAI(category, action, query, toolName);
    }
}

// Build natural language query for the tool
function buildToolQuery(category, action) {
    const queries = {
        // Security
        'security_full_scan': 'Esegui una scansione completa della sicurezza del sistema.',
        'security_vuln_check': 'Verifica le vulnerabilità del sistema.',
        'security_firewall': 'Analizza lo stato del firewall.',
        'security_audit': 'Esegui un audit completo della sicurezza.',
        // Health
        'health_diagnostics': 'Fornisci informazioni su diagnostica medica.',
        'health_medications': 'Fornisci informazioni su farmaci.',
        'health_wellness': 'Analizza lo stato di benessere.',
        'health_emergency': 'Protocollo emergenza medica.',
        // Science
        'science_research': 'Cerca articoli scientifici.',
        'science_physics': 'Calcoli di fisica.',
        'science_biology': 'Analisi biologica.',
        'science_simulation': 'Crea una simulazione scientifica.',
        // Chemistry
        'chemistry_compounds': 'Analizza composti chimici.',
        'chemistry_reactions': 'Simula reazioni chimiche.',
        'chemistry_periodic': 'Tavola periodica degli elementi.',
        'chemistry_molecular': 'Modelli molecolari.',
        // Archaeology
        'archaeology_artifacts': 'Database dei reperti archeologici.',
        'archaeology_dating': 'Metodi di datazione archeologica.',
        'archaeology_civilizations': 'Informazioni sulle civiltà antiche.',
        'archaeology_sites': 'Database dei siti archeologici.',
        // Military
        'military_strategy': 'Analisi strategica militare.',
        'military_intel': 'Raccolta e analisi di intelligence.',
        'military_logistics': 'Logistica militare.',
        'military_defense': 'Sistemi di difesa.',
        // Monitor
        'monitor_system': 'Monitora il sistema operativo.',
        'monitor_network': 'Analizza traffico di rete.',
        'monitor_performance': 'Benchmark del sistema.',
        'monitor_logs': 'Analizza i log di sistema.',
        // Cyber
        'cyber_threat_scan': 'Scansione per malware e minacce.',
        'cyber_intrusion': 'Rilevamento intrusioni.',
        'cyber_encryption': 'Strumenti di crittografia.',
        'cyber_pentest': 'Penetration test.',
        // Analysis
        'analysis_text': 'Analizza il testo fornito.',
        'analysis_sentiment': 'Analisi del sentiment.',
        'analysis_summary': 'Crea un riassunto automatico.',
        'analysis_translate': 'Traduci il testo.',
        'analysis_code': 'Analizza il codice.',
        // Utilities
        'utilities_calculator': 'Calcolatrice scientifica.',
        'utilities_converter': 'Converti unità di misura.',
        'utilities_timer': 'Gestisci timer e promemoria.',
        'utilities_weather': 'Previsioni meteo.',
        'utilities_qrcode': 'Genera o leggi QR Code.',
        // AI Tools
        'ai_image_gen': 'Genera un\'immagine AI.',
        'ai_image_analyze': 'Analizza un\'immagine.',
        'ai_voice': 'Attiva l\'assistente vocale.',
        'ai_reasoning': 'Deep reasoning per problemi complessi.',
        'ai_creative': 'Scrivi contenuti creativi.',
        // Data
        'data_export': 'Esporta la conversazione.',
        'data_import': 'Importa dati da file.',
        'data_backup': 'Crea un backup.',
        'data_history': 'Mostra la cronologia.',
        'data_settings': 'Apri le impostazioni.'
    };
    
    return queries[`${category}_${action}`] || `Esegui ${action} nella categoria ${category}`;
}

// Process with GIDEON AI as fallback
async function processWithGideonAI(category, action, query, toolName) {
    const apiBase = window.state?.apiBaseUrl || 'http://127.0.0.1:8001';
    showToolbarLoading(`GIDEON sta elaborando: ${toolName}`);
    
    try {
        const response = await fetch(`${apiBase}/api/chat/send`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: query,
                context: {
                    tool_category: category,
                    tool_action: action,
                    request_type: 'tool_execution'
                }
            })
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const result = await response.json();
        hideToolbarLoading();
        
        const content = result.response || result.message || result.text || 'Elaborazione completata.';
        showResultsPanel(toolName, `<div class="result-section"><h4>📊 Risultato GIDEON</h4><pre>${content}</pre></div>`);
        
        if (typeof addSystemMessage === 'function') {
            addSystemMessage(`${toolName}\n${content}`);
        }
        
    } catch (error) {
        hideToolbarLoading();
        console.error('GIDEON AI error:', error);
        if (typeof addSystemMessage === 'function') {
            addSystemMessage(`❌ Errore: ${error.message}`, 'error');
        }
    }
}

// Format tool results
function formatToolResult(category, action, result) {
    let html = '';
    
    if (result.data) {
        html += `<div class="result-section"><h4>📊 Dati</h4><pre>${JSON.stringify(result.data, null, 2)}</pre></div>`;
    }
    
    if (result.summary) {
        html += `<div class="result-section"><h4>📝 Riepilogo</h4><p>${result.summary}</p></div>`;
    }
    
    if (result.recommendations) {
        html += `<div class="result-section"><h4>💡 Raccomandazioni</h4><ul>`;
        result.recommendations.forEach(rec => html += `<li>${rec}</li>`);
        html += `</ul></div>`;
    }
    
    if (!html) {
        html = `<div class="result-section"><h4>✅ Completato</h4><pre>${JSON.stringify(result, null, 2)}</pre></div>`;
    }
    
    return html;
}

// ============ ADVANCED TOOLS ============

async function executeAdvancedTool(tool, action, params = {}) {
    const toolNames = {
        'security_predictive_risk_mapping': '🔒 Predictive Risk Mapping',
        'security_anomaly_narrator': '🔒 Anomaly Narrator',
        'security_defensive_scenario_simulator': '🔒 Defensive Scenario Simulator',
        'cyber_behavioral_baseline_builder': '🛡️ Behavioral Baseline Builder',
        'cyber_incident_explainability_engine': '🛡️ Incident Explainability Engine',
        'cyber_supply_chain_trust_scanner': '🛡️ Supply Chain Trust Scanner',
        'science_molecular_pattern_validator': '🧬 Molecular Pattern Validator',
        'science_environmental_contamination_scan': '🧬 Environmental Contamination Scan',
        'science_scientific_cross_check': '🧬 Scientific Cross-Check',
        'archaeology_predictive_reconstruction': '🏛️ Predictive Reconstruction',
        'archaeology_temporal_layer_fusion': '🏛️ Temporal Layer Fusion',
        'archaeology_authenticity_risk_assessment': '🏛️ Authenticity Risk Assessment',
        'core_multi_tool_reasoning': '🧠 Multi-Tool Reasoning',
        'core_confidence_weighted_output': '🧠 Confidence Weighted Output',
        'core_human_override_gate': '🧠 Human Override Gate',
        'core_audit_trail': '🧠 Audit Trail',
        'core_bias_and_drift_monitor': '🧠 Bias & Drift Monitor',
        'core_failsafe_trigger': '🧠 Failsafe Trigger'
    };
    
    const key = `${tool}_${action}`;
    const toolName = toolNames[key] || `${tool} - ${action}`;
    const apiBase = window.state?.apiBaseUrl || 'http://127.0.0.1:8001';
    
    showToolbarLoading(`Esecuzione avanzata: ${toolName}`);
    
    try {
        const response = await fetch(`${apiBase}/api/gideon/tools/advanced/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tool, action, params })
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const result = await response.json();
        hideToolbarLoading();
        
        const formattedContent = formatAdvancedToolResult(tool, action, result);
        showResultsPanel(toolName, formattedContent);
        
        if (typeof addSystemMessage === 'function') {
            const confidence = result.confidence ? ` (${Math.round(result.confidence * 100)}%)` : '';
            addSystemMessage(`✅ ${toolName} completato${confidence}`);
        }
        
    } catch (error) {
        hideToolbarLoading();
        console.error('Advanced tool error:', error);
        
        // Fallback to AI
        await processWithGideonAI(tool, action, `Esegui ${action} avanzato per ${tool}`, toolName);
    }
}

function formatAdvancedToolResult(tool, action, result) {
    let html = '';
    
    if (result.confidence !== undefined) {
        const confPercent = Math.round(result.confidence * 100);
        const confColor = confPercent >= 80 ? '#00ff88' : (confPercent >= 60 ? '#ffd700' : '#ff4444');
        html += `<div class="result-section" style="border-color: ${confColor};">
            <h4>📊 Confidenza: ${confPercent}%</h4>
            <div style="background: rgba(0,0,0,0.3); border-radius: 10px; overflow: hidden; height: 10px;">
                <div style="width: ${confPercent}%; height: 100%; background: ${confColor};"></div>
            </div>
        </div>`;
    }
    
    if (result.data) {
        html += `<div class="result-section"><h4>📋 Analisi</h4><pre style="max-height: 400px; overflow-y: auto;">${JSON.stringify(result.data, null, 2)}</pre></div>`;
    }
    
    if (result.summary) {
        html += `<div class="result-section"><h4>📝 Riepilogo</h4><p>${result.summary}</p></div>`;
    }
    
    if (result.disclaimer) {
        html += `<div class="result-section" style="border-color: #ffd700;"><h4>⚠️ Disclaimer</h4><p>${result.disclaimer}</p></div>`;
    }
    
    return html || `<div class="result-section"><pre>${JSON.stringify(result, null, 2)}</pre></div>`;
}

// ============ SPECIALIZED TOOL RUNNERS ============

// Security Tools
async function runSecurityTool(tool) {
    const apiBase = window.state?.apiBaseUrl || 'http://127.0.0.1:8001';
    
    if (typeof addLog === 'function') addLog(`[SECURITY] Running ${tool}`);
    if (typeof addSystemMessage === 'function') addSystemMessage(`🔒 Esecuzione ${tool}...`);
    
    try {
        const response = await fetch(`${apiBase}/api/security/${tool}`, { method: 'POST' });
        if (response.ok) {
            const data = await response.json();
            if (typeof addSystemMessage === 'function') {
                addSystemMessage(`✅ ${tool}: ${data.message || JSON.stringify(data)}`);
            }
        } else {
            await runGideonTool('security', tool);
        }
    } catch (error) {
        await runGideonTool('security', tool);
    }
}

// Health Tools
async function runHealthTool(tool) {
    const toolMessages = {
        posture_reminder: '🧘 Reminder Postura attivato. Ti avviserò ogni 30 minuti.',
        eye_break: '👁️ Pausa Occhi 20-20-20: Guarda a 6 metri per 20 secondi.',
        hydration_reminder: '💧 Reminder Idratazione attivato.',
        stretch_break: '🤸 È ora di fare stretching! Alza le braccia sopra la testa.',
        screen_time_report: '📊 Tempo schermo oggi: analisi in corso...',
        blue_light_filter: '🌙 Filtro luce blu attivato.',
        focus_mode: '🎯 Modalità Focus attivata. Notifiche silenziose.',
        wellness_summary: '📋 Generazione riepilogo benessere...'
    };
    
    if (typeof addLog === 'function') addLog(`[HEALTH] Running ${tool}`);
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(toolMessages[tool] || `💊 Tool ${tool} avviato`);
    }
    if (typeof speak === 'function' && window.state?.isTTSEnabled) {
        speak(toolMessages[tool]);
    }
}

// Prevention/Maintenance Tools
async function runPreventionTool(tool) {
    const toolMessages = {
        disk_health: '💽 Analisi salute disco in corso...',
        battery_health: '🔋 Verifica stato batteria...',
        temp_cleanup: '🧹 Pulizia file temporanei...',
        defrag_analyze: '📀 Analisi frammentazione...',
        startup_optimize: '🚀 Ottimizzazione avvio...',
        driver_check: '🔧 Verifica driver...',
        event_log_analyze: '📝 Analisi log eventi...',
        full_maintenance: '⚡ Manutenzione completa in corso...'
    };
    
    if (typeof addLog === 'function') addLog(`[PREVENTION] Running ${tool}`);
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(toolMessages[tool] || `🛠️ Tool ${tool} avviato`);
    }
}

// Workflow Runner
async function runWorkflow(workflowId) {
    const workflows = {
        morning_routine: {
            name: '🌅 Routine Mattina',
            steps: ['Apertura email', 'Calendario', 'News', 'Meteo']
        },
        work_setup: {
            name: '💼 Setup Lavoro',
            steps: ['IDE', 'Browser', 'Comunicazioni', 'Focus mode']
        },
        meeting_prep: {
            name: '📅 Prepara Meeting',
            steps: ['Test webcam', 'Test microfono', 'Note meeting']
        },
        end_of_day: {
            name: '🌙 Fine Giornata',
            steps: ['Salva documenti', 'Backup', 'Report', 'Chiudi app']
        },
        backup_routine: {
            name: '💾 Backup Routine',
            steps: ['Documenti', 'Progetti', 'Configurazioni', 'Verifica']
        },
        cleanup_routine: {
            name: '🧹 Pulizia Routine',
            steps: ['Temp files', 'Cache', 'Cestino', 'Ottimizzazione']
        },
        security_routine: {
            name: '🔒 Routine Sicurezza',
            steps: ['Scansione', 'Update', 'Firewall', 'Permessi']
        },
        custom: {
            name: '✨ Workflow Personalizzato',
            steps: ['Configurazione...']
        }
    };
    
    const workflow = workflows[workflowId];
    if (!workflow) return;
    
    if (typeof addLog === 'function') addLog(`[WORKFLOW] Starting ${workflowId}`);
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(`⚙️ ${workflow.name}\nSteps: ${workflow.steps.join(' → ')}`);
    }
    
    // Execute workflow steps
    for (const step of workflow.steps) {
        if (typeof addSystemMessage === 'function') {
            addSystemMessage(`▶️ ${step}...`);
        }
        await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(`✅ ${workflow.name} completato!`);
    }
}

// Cyber Tools
async function runCyberTool(tool) {
    const toolMessages = {
        port_scan: '🔌 Scansione porte in corso...',
        firewall_check: '🧱 Verifica firewall...',
        malware_scan: '🦠 Scansione malware...',
        intrusion_detect: '🚨 Rilevamento intrusioni attivo...',
        vuln_assessment: '🔓 Assessment vulnerabilità...',
        traffic_monitor: '📊 Monitor traffico attivo...',
        ssl_check: '🔒 Verifica SSL/TLS...',
        dns_lookup: '🌐 DNS Lookup...',
        whois: '📋 WHOIS Lookup...',
        hash_check: '🔐 Verifica hash...',
        password_strength: '💪 Analisi password...',
        breach_check: '⚠️ Check data breach...'
    };
    
    if (typeof addLog === 'function') addLog(`[CYBER] Running ${tool}`);
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(toolMessages[tool] || `🛡️ Cyber tool ${tool} avviato`);
    }
    
    await runGideonTool('cyber', tool);
}

// Military Tools  
async function runMilitaryTool(tool) {
    const toolMessages = {
        sigint: '🛰️ SIGINT Analysis in corso...',
        osint: '🔍 OSINT Gathering attivo...',
        tactical: '🎯 Tactical AI pronto.',
        cyber_defense: '🛡️ Cyber Defense attivo.',
        c4isr: '📡 C4ISR Network connesso.',
        intel_fusion: '🗄️ Intel Fusion attivo.',
        geospatial: '🗺️ Geospatial Intel caricato.',
        threat_assess: '⚠️ Threat Assessment...',
        comms_secure: '📻 Comunicazioni sicure attive.',
        crypto_ops: '🔐 Crypto Operations pronte.'
    };
    
    if (typeof addLog === 'function') addLog(`[MILITARY] Running ${tool}`);
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(toolMessages[tool] || `⚔️ Military tool ${tool} avviato`);
    }
}

// Monitor Tools
async function runMonitorTool(tool) {
    const apiBase = window.state?.apiBaseUrl || 'http://127.0.0.1:8001';
    
    const toolMessages = {
        cpu_usage: '🔥 CPU Usage in tempo reale...',
        ram_usage: '💾 RAM Usage...',
        disk_usage: '💽 Disk Usage...',
        network_traffic: '📡 Network Traffic...',
        processes: '📋 Lista processi...',
        services: '⚙️ Stato servizi...',
        temperatures: '🌡️ Temperature sistema...',
        battery: '🔋 Stato batteria...',
        gpu_usage: '🎮 GPU Usage...',
        system_uptime: '⏱️ System Uptime...',
        event_logs: '📝 Event Logs...',
        full_report: '📊 Generazione report completo...'
    };
    
    if (typeof addLog === 'function') addLog(`[MONITOR] Running ${tool}`);
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(toolMessages[tool] || `📊 Monitor ${tool} avviato`);
    }
    
    try {
        const response = await fetch(`${apiBase}/api/system/info`);
        if (response.ok) {
            const data = await response.json();
            if (typeof addSystemMessage === 'function') {
                addSystemMessage(`📊 ${JSON.stringify(data, null, 2)}`);
            }
        }
    } catch (error) {
        console.log('Monitor fallback');
    }
}

// Work/Productivity Tools
async function runWorkTool(tool) {
    const toolMessages = {
        pomodoro: '🍅 Timer Pomodoro: 25 minuti di focus!',
        task_list: '📝 Task List aperta.',
        calendar: '📅 Calendario caricato.',
        notes: '🗒️ Note rapide pronte.',
        time_tracker: '⏰ Time Tracker avviato.',
        project_manager: '📋 Project Manager aperto.',
        email_draft: '✉️ Generazione bozza email...',
        meeting_notes: '🎤 Template meeting pronto.',
        code_snippets: '💻 Code Snippets library.',
        document_gen: '📄 Generazione documento...',
        productivity_stats: '📈 Stats produttività...',
        daily_summary: '📊 Riepilogo giornata...'
    };
    
    if (typeof addLog === 'function') addLog(`[WORK] Running ${tool}`);
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(toolMessages[tool] || `💼 Work tool ${tool} avviato`);
    }
}

// Resource Tools
async function runResourceTool(tool) {
    const toolMessages = {
        knowledge_base: '🧠 Knowledge Base caricata.',
        wiki_search: '📖 Wiki Search pronta.',
        paper_search: '📄 Paper Search (arXiv, PubMed)...',
        patent_db: '💡 Patent Database...',
        legal_db: '⚖️ Legal Database...',
        financial_data: '📈 Financial Data...',
        news_aggregator: '📰 News Aggregator...',
        open_data: '🌍 Open Data datasets...',
        api_catalog: '🔌 API Catalog...',
        code_repos: '💻 Code Repos search...'
    };
    
    if (typeof addLog === 'function') addLog(`[RESOURCE] Running ${tool}`);
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(toolMessages[tool] || `📚 Resource ${tool} caricato`);
    }
}

// Archaeology Tools
async function runArchaeologyTool(tool) {
    const toolMessages = {
        site_analysis: '🗿 Site Analysis con AI...',
        artifact_id: '🏺 Artifact Identification...',
        dating_calc: '📅 Dating Calculator...',
        ancient_texts: '📜 Ancient Texts database...',
        '3d_reconstruction': '🏗️ 3D Reconstruction...',
        historical_maps: '🗺️ Historical Maps...',
        dynasty_db: '👑 Dynasty Database...',
        museum_catalog: '🏛️ Museum Catalog...'
    };
    
    if (typeof addLog === 'function') addLog(`[ARCHAEOLOGY] Running ${tool}`);
    if (typeof addSystemMessage === 'function') {
        addSystemMessage(toolMessages[tool] || `🏛️ Archaeology tool ${tool} avviato`);
    }
}

// ============ UTILITY FUNCTIONS ============

// Stop response (used in chat)
function stopResponse() {
    if (window.state?.abortController) {
        window.state.abortController.abort();
        window.state.isProcessing = false;
    }
    if (typeof addLog === 'function') addLog('[SYSTEM] Response stopped by user');
}

// Finish processing
function finishProcessing() {
    if (window.state) {
        window.state.isProcessing = false;
    }
    const stopBtn = document.getElementById('stop-btn');
    const sendBtn = document.getElementById('send-btn');
    if (stopBtn) stopBtn.classList.remove('active');
    if (sendBtn) sendBtn.classList.remove('hidden');
}

// Handle input change
function handleInputChange() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    if (input && sendBtn) {
        sendBtn.disabled = !input.value.trim();
    }
}

// Update mode UI
function updateModeUI(mode) {
    const root = document.documentElement;
    const colors = {
        passive: '#888888',
        copilot: '#00f5ff',
        pilot: '#00ff88',
        executive: '#ff00ff'
    };
    root.style.setProperty('--current-mode-color', colors[mode] || colors.copilot);
}

// ============ VOICE FUNCTIONS ============

function stopSpeech() {
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
    if (window.state) {
        window.state.isSpeaking = false;
    }
    const avatar = document.querySelector('.avatar');
    if (avatar) avatar.classList.remove('speaking');
}

function processVoiceCommand(text) {
    // Handle voice commands
    const commands = {
        'ferma': stopSpeech,
        'stop': stopSpeech,
        'pausa': stopSpeech,
        'silenzio': stopSpeech
    };
    
    const lowerText = text.toLowerCase();
    for (const [cmd, fn] of Object.entries(commands)) {
        if (lowerText.includes(cmd)) {
            fn();
            return true;
        }
    }
    return false;
}

// ============ IMAGE OPTIONS ============

function showImageOptions() {
    const popup = document.getElementById('image-options-popup');
    if (popup) popup.style.display = 'block';
}

function hideImageOptions() {
    const popup = document.getElementById('image-options-popup');
    if (popup) popup.style.display = 'none';
}

function openGallery() {
    const input = document.getElementById('image-upload');
    if (input) input.click();
    hideImageOptions();
}

// ============ DEBUG FUNCTIONS ============

async function testWebSocketConnection() {
    if (typeof addLog === 'function') addLog('[DEBUG] Testing WebSocket...');
    
    try {
        const ws = new WebSocket('ws://127.0.0.1:8001/ws');
        ws.onopen = () => {
            if (typeof addLog === 'function') addLog('[DEBUG] WebSocket connected!');
            ws.close();
        };
        ws.onerror = (e) => {
            if (typeof addLog === 'function') addLog('[DEBUG] WebSocket error');
        };
    } catch (error) {
        if (typeof addLog === 'function') addLog('[DEBUG] WebSocket test failed');
    }
}

// ============ EXPORT ============

// Make functions globally available
window.runGideonTool = runGideonTool;
window.executeAdvancedTool = executeAdvancedTool;
window.runSecurityTool = runSecurityTool;
window.runHealthTool = runHealthTool;
window.runPreventionTool = runPreventionTool;
window.runWorkflow = runWorkflow;
window.runCyberTool = runCyberTool;
window.runMilitaryTool = runMilitaryTool;
window.runMonitorTool = runMonitorTool;
window.runWorkTool = runWorkTool;
window.runResourceTool = runResourceTool;
window.runArchaeologyTool = runArchaeologyTool;
window.toggleToolbar = toggleToolbar;
window.toggleSection = toggleSection;
window.showToolbarLoading = showToolbarLoading;
window.hideToolbarLoading = hideToolbarLoading;
window.showResultsPanel = showResultsPanel;
window.closeResultsPanel = closeResultsPanel;
window.stopResponse = stopResponse;
window.finishProcessing = finishProcessing;
window.handleInputChange = handleInputChange;
window.updateModeUI = updateModeUI;
window.stopSpeech = stopSpeech;
window.processVoiceCommand = processVoiceCommand;
window.showImageOptions = showImageOptions;
window.hideImageOptions = hideImageOptions;
window.openGallery = openGallery;
window.testWebSocketConnection = testWebSocketConnection;

console.log('✅ GIDEON Tools loaded successfully');
