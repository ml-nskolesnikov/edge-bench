// Internationalization (i18n) for Edge-Bench

const translations = {
    ru: {
        // Navigation
        'Dashboard': 'Главная',
        'Devices': 'Устройства',
        'Models': 'Модели',
        'Experiments': 'Эксперименты',
        'Results': 'Результаты',
        'Compare': 'Сравнение',
        'Dependencies': 'Зависимости',
        'Benchmark': 'Бенчмарк',
        '+ New': '+ Новый',

        // Dashboard
        'Recent Experiments': 'Последние эксперименты',
        'Quick Actions': 'Быстрые действия',
        'Export CSV': 'Экспорт CSV',
        'Export JSON': 'Экспорт JSON',
        'Avg Latency': 'Ср. задержка',
        'View all': 'Все',
        'Manage': 'Управление',

        // Devices
        'Registered Devices': 'Зарегистрированные устройства',
        'Add Device': 'Добавить устройство',
        'Name': 'Название',
        'IP Address': 'IP адрес',
        'Port': 'Порт',
        'Status': 'Статус',
        'Version': 'Версия',
        'Last Seen': 'Последняя активность',
        'Actions': 'Действия',
        'Ping': 'Проверить',
        'Update': 'Обновить',
        'Delete': 'Удалить',
        'Description': 'Описание',
        'Cancel': 'Отмена',
        'Add': 'Добавить',
        'Installation': 'Установка',
        'To install the agent on a Raspberry Pi:': 'Для установки агента на Raspberry Pi:',
        'online': 'онлайн',
        'offline': 'оффлайн',
        'busy': 'занят',
        'Check Version': 'Проверить версию',
        'Update Agent': 'Обновить агент',
        'View Models': 'Посмотреть модели',
        'No devices yet': 'Устройств пока нет',
        'Add a device above or install agent on RPi': 'Добавьте устройство выше или установите агент на RPi',

        // Models
        'Model Repository': 'Репозиторий моделей',
        'Upload New Model': 'Загрузить модель',
        'Available Models': 'Доступные модели',
        'Model File': 'Файл модели',
        'Quantization Type': 'Тип квантизации',
        'Auto-detect': 'Автоопределение',
        'Upload': 'Загрузить',
        'Refresh': 'Обновить',
        'Size': 'Размер',
        'Quantization': 'Квантизация',
        'Hash': 'Хэш',
        'Uploaded': 'Загружен',
        'Deploy to Device': 'Развернуть на устройство',
        'Deploy': 'Развернуть',
        'Download': 'Скачать',
        'Models on Devices': 'Модели на устройствах',
        'Upload Model to Server': 'Загрузить модель на сервер',
        'Select device...': 'Выберите устройство...',
        'Select model...': 'Выберите модель...',
        'No models': 'Нет моделей',
        'Loading...': 'Загрузка...',

        // Experiments
        'All Experiments': 'Все эксперименты',
        'New Experiment': 'Новый эксперимент',
        'Model': 'Модель',
        'Device': 'Устройство',
        'Created': 'Создан',
        'View': 'Просмотр',
        'Rerun': 'Повторить',
        'queued': 'в очереди',
        'running': 'выполняется',
        'completed': 'завершён',
        'failed': 'ошибка',
        'cancelled': 'отменён',
        'pending': 'ожидание',

        // New Experiment
        'Model Source': 'Источник модели',
        'Model on device': 'Модель на устройстве',
        'Model on Device': 'Модель на устройстве',
        'From repository': 'Из репозитория',
        'From repository (deploy first)': 'Из репозитория (сначала развернуть)',
        'Model from Repository': 'Модель из репозитория',
        'Manual path': 'Указать путь вручную',
        'Model Path': 'Путь к модели',
        'Backend': 'Бэкенд',
        'Batch Size': 'Размер батча',
        'Threads': 'Потоки',
        'Warmup Runs': 'Прогревочные запуски',
        'Benchmark Runs': 'Тестовые запуски',
        'Create & Run': 'Создать и запустить',
        'Batch Experiment': 'Пакетный эксперимент',
        'To run multiple models at once, use the API:': 'Для запуска нескольких моделей сразу используйте API:',
        'Select device first...': 'Сначала выберите устройство...',
        'Models already deployed to the selected device': 'Модели, уже загруженные на выбранное устройство',
        'Model will be deployed to device before experiment': 'Модель будет загружена на устройство перед экспериментом',
        'Full path to model on the Raspberry Pi': 'Полный путь к модели на Raspberry Pi',

        // Experiment Detail
        'Experiment Results': 'Результаты эксперимента',
        'Mean Latency': 'Средняя задержка',
        'Throughput': 'Пропускная способность',
        'Model Load': 'Загрузка модели',
        'CPU Temp': 'Температура CPU',
        'Latency Distribution': 'Распределение задержки',
        'Mean': 'Среднее',
        'Std': 'Стд. откл.',
        'System Metrics': 'Системные метрики',
        'CPU Usage': 'Загрузка CPU',
        'Memory': 'Память',
        'Model Info': 'Информация о модели',
        'Device Info': 'Информация об устройстве',
        'Test Parameters': 'Параметры теста',
        'Hostname': 'Имя хоста',
        'Platform': 'Платформа',
        'CPUs': 'Ядра CPU',
        'Execution Logs': 'Логи выполнения',
        'Rerun Experiment': 'Повторить эксперимент',
        'Back to List': 'Назад к списку',
        'Error': 'Ошибка',
        'First inference': 'Первый инференс',
        'Duration': 'Длительность',
        'Experiment not found.': 'Эксперимент не найден.',

        // Results
        'Benchmark Results': 'Результаты бенчмарков',
        'Experiment': 'Эксперимент',
        'Latency (mean)': 'Задержка (средн.)',
        'Latency (p95)': 'Задержка (p95)',
        'FPS': 'FPS',
        'No results yet. Run some experiments first.': 'Результатов пока нет. Сначала запустите эксперименты.',

        // Common
        'No experiments yet': 'Пока нет экспериментов',
        'Create one': 'Создать',
        'No devices registered': 'Нет зарегистрированных устройств',
        'No models uploaded yet': 'Модели ещё не загружены',
        'No results yet': 'Пока нет результатов',
        'Yes': 'Да',
        'No': 'Нет',
        'Close': 'Закрыть',
        'Save': 'Сохранить',
        'unknown': 'неизвестно',
        'ID': 'ID',
    },
    en: {
        // English is default, no translations needed
    }
};

// Get current language from localStorage or default to Russian
function getLang() {
    return localStorage.getItem('edgebench_lang') || 'ru';
}

// Set language
function setLang(lang) {
    localStorage.setItem('edgebench_lang', lang);
    location.reload();
}

// Translate text
function t(text) {
    const lang = getLang();
    if (lang === 'en') return text;
    return translations.ru[text] || text;
}

// Translate page on load
function translatePage() {
    const lang = getLang();
    if (lang === 'en') return;

    // Translate text nodes
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (translations.ru[key]) {
            el.textContent = translations.ru[key];
        }
    });

    // Translate placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (translations.ru[key]) {
            el.placeholder = translations.ru[key];
        }
    });

    // Translate all text content that matches
    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );

    while (walker.nextNode()) {
        const node = walker.currentNode;
        const text = node.textContent.trim();
        if (text && translations.ru[text]) {
            node.textContent = node.textContent.replace(text, translations.ru[text]);
        }
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', translatePage);
