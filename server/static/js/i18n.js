// Internationalization (i18n) for Edge-Bench
// English is the base language. translations.ru maps EN keys → RU values.
// In EN mode a reverse map (RU → EN) is built at runtime for text-node translation.

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
        'Schedules': 'Расписания',
        'Settings': 'Настройки',
        'footer-text': 'Edge-Bench v1.0 | Бенчмаркинг ML на Edge устройствах',

        // Common UI
        'All': 'Все',
        'Yes': 'Да',
        'No': 'Нет',
        'Add': 'Добавить',
        'Save': 'Сохранить',
        'Cancel': 'Отмена',
        'Close': 'Закрыть',
        'Delete': 'Удалить',
        'Update': 'Обновить',
        'Upload': 'Загрузить',
        'Download': 'Скачать',
        'Deploy': 'Развернуть',
        'Refresh': 'Обновить',
        'Reset': 'Сбросить',
        'Search': 'Поиск',
        'Loading...': 'Загрузка...',
        'Checking...': 'Проверка...',
        'Deploying...': 'Развертывание...',
        'Ready': 'Готово',
        'Active': 'Активно',
        'ID': 'ID',
        'Name': 'Название',
        'Status': 'Статус',
        'Type': 'Тип',
        'Size': 'Размер',
        'Path': 'Путь',
        'Date': 'Дата',
        'Version': 'Версия',
        'Description': 'Описание',
        'Actions': 'Действия',
        'Port': 'Порт',
        'Total': 'Всего',
        'Disk': 'Диск',
        'Memory': 'Память',
        'Temperature': 'Температура',
        'unknown': 'неизвестно',
        'Error': 'Ошибка',
        'View': 'Просмотр',
        'Manage': 'Управление',
        'Create': 'Создать',
        'Run': 'Запуск',

        // Status values — lowercase (used in JS / API responses)
        'queued': 'в очереди',
        'running': 'выполняется',
        'completed': 'завершён',
        'failed': 'ошибка',
        'cancelled': 'отменён',
        'pending': 'ожидание',
        'online': 'онлайн',
        'offline': 'оффлайн',
        'busy': 'занят',

        // Status values — sentence-case (filter dropdowns)
        'Queued': 'В очереди',
        'Running': 'Выполняется',
        'Completed': 'Завершён',
        'Failed': 'Ошибка',
        'Cancelled': 'Отменён',

        // Status values — ALL CAPS (Jinja status badges)
        'QUEUED': 'В ОЧЕРЕДИ',
        'RUNNING': 'ВЫПОЛНЯЕТСЯ',
        'COMPLETED': 'ЗАВЕРШЁН',
        'FAILED': 'ОШИБКА',
        'CANCELLED': 'ОТМЕНЁН',

        // Dashboard
        'Recent Experiments': 'Последние эксперименты',
        'Quick Actions': 'Быстрые действия',
        'Export CSV': 'Экспорт CSV',
        'Export JSON': 'Экспорт JSON',
        'Avg Latency': 'Ср. задержка',
        'View all': 'Все',
        'Create first experiment': 'Создать первый эксперимент',
        'No experiments yet': 'Пока нет экспериментов',

        // Devices
        'Registered Devices': 'Зарегистрированные устройства',
        'Add Device': 'Добавить устройство',
        'IP Address': 'IP адрес',
        'Last Seen': 'Последняя активность',
        'Agent version': 'Версия агента',
        'Latency (ping)': 'Задержка (ping)',
        'Ping': 'Пинг',
        'Check Version': 'Проверить версию',
        'Update Agent': 'Обновить агент',
        'View Models': 'Посмотреть модели',
        'No devices yet': 'Устройств пока нет',
        'No devices': 'Нет устройств',
        'Add a device above or install agent on RPi': 'Добавьте устройство выше или установите агент на RPi',
        'Manage connected Edge devices': 'Управление подключёнными Edge-устройствами',
        'Checking connection with devices...': 'Проверка связи с устройствами...',
        'Page loaded, polling': 'Страница загружена, идет опрос',
        'Installation': 'Установка',
        'Agent installation': 'Установка агента',
        'To install the agent on a Raspberry Pi:': 'Для установки агента на Raspberry Pi:',
        'For removal:': 'Для удаления:',
        'No models on device': 'Нет моделей на устройстве',
        'Deploy model from server': 'Развернуть модель с сервера',
        'Load to device': 'Загрузить на устройство',
        'Loading models...': 'Загрузка моделей...',
        'Loading devices...': 'Загрузка устройств...',
        'Device offline': 'Устройство оффлайн',
        'Device unavailable': 'Устройство недоступно',
        'Add connection data for Edge device': 'Укажите данные для подключения к Edge-устройству',

        // Models
        'Model Repository': 'Репозиторий моделей',
        'Upload New Model': 'Загрузить модель',
        'Available Models': 'Доступные модели',
        'Model File': 'Файл модели',
        'Quantization Type': 'Тип квантизации',
        'Auto-detect': 'Автоопределение',
        'Quantization': 'Квантизация',
        'Hash': 'Хэш',
        'Uploaded': 'Загружен',
        'Deploy to Device': 'Развернуть на устройство',
        'Models on Devices': 'Модели на устройствах',
        'Upload Model to Server': 'Загрузить модель на сервер',
        'Select device...': 'Выберите устройство...',
        'Select model...': 'Выберите модель...',
        'Select device': 'Выберите устройство',
        'Select model': 'Выберите модель',
        'No models': 'Нет моделей',
        'No models uploaded yet': 'Модели ещё не загружены',
        'Upload first model via the form above': 'Загрузите первую модель через форму выше',
        'Upload and manage TFLite models for benchmarking': 'Загружайте и управляйте TFLite моделями для бенчмаркинга',
        'Select .tflite file to upload to server': 'Выберите .tflite файл для загрузки на сервер',
        'Model name': 'Имя модели',
        'Model metrics': 'Метрики модели',
        'Already on device': 'Уже на устройстве',

        // Experiments
        'All Experiments': 'Все эксперименты',
        'New Experiment': 'Новый эксперимент',
        'Model': 'Модель',
        'Device': 'Устройство',
        'Created': 'Создан',
        'Rerun': 'Повторить',
        'Backend': 'Бэкенд',
        'Backends': 'Бэкенды',
        'Search by name, model, ID...': 'Поиск по названию, модели, ID...',
        'Task queue': 'Очередь задач',
        'Currently running:': 'Сейчас выполняется:',
        'Retry all failed': 'Повторить все неудавшиеся',
        'Delete by status': 'Удалить по статусу',
        'Select all': 'Выбрать все',
        'selected': 'выбрано',
        'selected for comparison': 'выбрано для сравнения',
        'experiments will be created': 'экспериментов будет создано',
        'Creating experiments...': 'Создание экспериментов...',
        'Manage and analyze ML model benchmarks on Edge devices': 'Управление и анализ бенчмарков ML-моделей на Edge-устройствах',
        'Try changing search parameters or resetting filters': 'Попробуйте изменить параметры поиска или сбросить фильтры',
        'History cleared': 'История очищена',
        'Experiment': 'Эксперимент',
        'Experiments': 'Экспериментов',

        // New Experiment
        'Model Source': 'Источник модели',
        'Model on device': 'Модель на устройстве',
        'Model on Device': 'Модель на устройстве',
        'From repository': 'Из репозитория',
        'From repository (deploy first)': 'Из репозитория (сначала развернуть)',
        'Model from Repository': 'Модель из репозитория',
        'Manual path': 'Вручную',
        'Model Path': 'Путь к модели',
        'Batch Size': 'Размер батча',
        'Threads': 'Потоки',
        'CPU threads': 'Потоки (CPU)',
        'Warmup Runs': 'Прогревочные запуски',
        'Benchmark Runs': 'Тестовые запуски',
        'Create & Run': 'Создать и запустить',
        'Batch Experiment': 'Пакетный эксперимент',
        'To run multiple models at once, use the API:': 'Для запуска нескольких моделей сразу используйте API:',
        'Select device first...': 'Сначала выберите устройство...',
        'Models already deployed to the selected device': 'Модели, уже загруженные на выбранное устройство',
        'Model will be deployed to device before experiment': 'Модель будет загружена на устройство перед экспериментом',
        'Model will be automatically deployed to device': 'Модель будет автоматически загружена на устройство',
        'Full path to model on the Raspberry Pi': 'Полный путь к модели на Raspberry Pi',
        'Path to model on device': 'Путь к модели на устройстве',
        'Experiment name in MLflow (default: edge-bench)': 'Имя эксперимента в MLflow (по умолчанию: edge-bench)',
        'Integrations (MLflow / W&B)': 'Интеграции (MLflow / W&B)',
        'Results will be automatically logged to MLflow. Leave fields empty to disable.': 'Результаты экспериментов будут автоматически логироваться в MLflow. Оставьте поля пустыми, чтобы отключить.',
        'Use a descriptive name to identify results': 'Используйте понятное название для идентификации результатов',

        // Experiment Detail
        'Experiment Results': 'Результаты эксперимента',
        'Mean Latency': 'Средняя задержка',
        'Average latency': 'Средняя задержка',
        'Throughput': 'Пропускная способность',
        'Model Load': 'Загрузка модели',
        'CPU Temp': 'Температура CPU',
        'Latency Distribution': 'Распределение задержки',
        'Mean': 'Среднее',
        'Std': 'Стд. откл.',
        'System Metrics': 'Системные метрики',
        'CPU Usage': 'Загрузка CPU',
        'Model Info': 'Информация о модели',
        'Device Info': 'Информация об устройстве',
        'Test Parameters': 'Параметры теста',
        'Hostname': 'Имя хоста',
        'Platform': 'Платформа',
        'CPUs': 'Ядра CPU',
        'Execution Logs': 'Логи выполнения',
        'Execution history': 'История выполнения',
        'Rerun Experiment': 'Повторить эксперимент',
        'Back to List': 'Назад к списку',
        'First inference': 'Первый инференс',
        'Duration': 'Длительность',
        'Experiment not found.': 'Эксперимент не найден.',
        'Batch': 'Батч',
        'Warmup': 'Прогрев',
        'Iterations': 'Итераций',
        'runs': 'запусков',
        'sec': 'сек',
        'current': 'текущее',
        'value': 'значение',
        'models selected': 'моделей выбрано',

        // Results
        'Benchmark Results': 'Результаты бенчмарков',
        'Latency (mean)': 'Задержка (средн.)',
        'Latency (p95)': 'Задержка (p95)',
        'Latency (ms)': 'Латентность (ms)',
        'Throughput (FPS)': 'Пропускная способность (FPS)',
        'FPS': 'FPS',
        'No results yet': 'Пока нет результатов',
        'No results yet. Run some experiments first.': 'Результатов пока нет. Сначала запустите эксперименты.',
        'No devices registered': 'Нет зарегистрированных устройств',

        // Benchmark page
        'Run performance tests of models on Edge devices': 'Запускайте тесты производительности моделей на Edge-устройствах',
        'Quick run': 'Быстрый запуск',
        'Quick benchmark run': 'Быстрый запуск',
        'Detailed test of a single model with full statistics': 'Детальный тест одной модели с полной статистикой',
        'Multiple tests at once': 'Множество тестов за один раз',
        'Batch benchmark': 'Пакетный бенчмарк',
        'Run benchmark scripts': 'Запуск бенчмарков',
        'Run custom scripts for testing models': 'Запуск пользовательских скриптов для тестирования моделей',
        'Run script': 'Запуск скрипта',
        'Upload script': 'Загрузить скрипт',
        'View script': 'Просмотр скрипта',
        'Available scripts': 'Доступные скрипты',
        'No scripts yet. Upload the first script above.': 'Скриптов пока нет. Загрузите первый скрипт выше.',
        'Benchmark is running... This may take a few minutes.': 'Выполняется бенчмарк... Это может занять несколько минут.',
        'Best latency': 'Лучшая латентность',
        'Max throughput': 'Макс. пропускная способность',
        'Model directory': 'Директория моделей',
        'Automatic test of all models in directory': 'Автоматический тест всех моделей в директории',
        'Run Python code directly on device without uploading a file': 'Запустите Python код напрямую на устройстве без загрузки файла',
        'Reproducible benchmark for ECCV2026 paper': 'Воспроизводимый бенчмарк для ECCV2026 paper',
        'Detailed JSON report': 'Подробный JSON-отчёт',
        'Download JSON': 'Скачать JSON',
        'Check and run': 'Проверьте и запустите',
        'Storage paths': 'Пути хранения',
        'Architectures': 'Архитектуры',
        'Strategies': 'Стратегии',
        'Math library': 'Математическая библиотека',
        'Package (pip)': 'Пакет (pip)',
        'Database': 'База данных',

        // Compare page
        'Compare results': 'Сравнить результаты',
        'Comparison results': 'Сравнение результатов',
        'Select experiments for visual performance comparison': 'Выберите эксперименты для визуального сравнения производительности',
        'Detailed comparison': 'Детальное сравнение',
        'Loading comparison...': 'Загружаю сравнение...',
        'Best / worst difference': 'Разница лучший / худший',
        'Change device': 'Сменить устройство',
        'Select device for rerun:': 'Выберите устройство для повторного запуска эксперимента:',
        'add devices': 'добавьте устройства',

        // Schedules
        'Automatic benchmark runs by cron (UTC)': 'Автоматические запуски бенчмарков по cron (UTC)',
        'Schedule': 'Расписание',
        'Next run': 'Следующий запуск',
        'Last run': 'Последний запуск',
        'Create schedule': 'Создать расписание',
        'All schedules': 'Все расписания',
        'No schedules yet': 'Расписаний ещё нет',
        'Create first schedule': 'Создать первое расписание',
        'Add first schedule and run experiments automatically.': 'Добавьте первое расписание и запускайте эксперименты автоматически.',
        'Upcoming runs': 'Ближайшие запуски',
        'Every hour': 'Каждый час',
        'Every 6 hours': 'Каждые 6 часов',
        'Every night at 2:00': 'Каждую ночь в 2:00',

        // Settings
        'Edge-Bench server configuration': 'Конфигурация Edge-Bench сервера',
        'Server parameters': 'Параметры сервера',
        'Server port': 'Порт сервера',
        'Agent timeout (sec)': 'Таймаут агента (сек)',
        'Task timeout (sec)': 'Таймаут задачи (сек)',
        'Max parallel tasks': 'Макс. параллельных задач',
        'Task timeout': 'Таймаут задач',
        'Max tasks': 'Макс. задач',
        'Agent connection timeout': 'Таймаут подключения к агенту на устройстве',
        'Max simultaneous experiments': 'Количество одновременно выполняемых экспериментов',
        'Max execution time per experiment': 'Максимальное время выполнения одного эксперимента',

        // Dependencies
        'Check dependencies': 'Проверить зависимости',
        'Dependency status': 'Состояние зависимостей',
        'Dependencies not configured': 'Зависимости не настроены',
        'Dependencies required for agent operation on devices. Critical ones are marked when missing.': 'Зависимости, необходимые для работы агента на устройствах. Критические отмечены при отсутствии.',
        'Manage dependencies': 'Управление зависимостями',
        'Devices and dependencies': 'Устройства и зависимости',
        'Checking dependencies...': 'Проверка зависимостей...',
        'Checking dependencies on device...': 'Проверяем зависимостей на устройстве...',
        'Check on device:': 'Проверить на устройстве:',
        'Add dependency': 'Добавить зависимость',
        'Add first': 'Добавить первую',
        'Installed': 'Установлено',
        'Verified': 'Проверено',
        'Check dependencies': 'Проверка зависимостей',

        // Mass deploy
        'Mass deployment': 'Массовое развертывание',
        'Select status for bulk delete:': 'Выберите статус для массового удаления:',

        // Schedules
        'New Schedule': 'Новое расписание',
        'Create Schedule': 'Создать расписание',
        'Create first schedule': 'Создать первое расписание',
        'Enabled': 'Активно',
        'Disabled': 'Отключено',
        'Benchmark runs': 'Тестовые запуски',
        'Warmup runs': 'Прогревочные запуски',

        // Buttons with prefix characters (exact text-node match needed)
        '+ Add Device': '+ Добавить устройство',
        '+ New Experiment': '+ Новый эксперимент',
        '+ Create Schedule': '+ Создать расписание',
        '+ Add': '+ Добавить',
        '+ Add first': '+ Добавить первую',
        'Create Experiment': 'Создать эксперимент',
        'Download JSON': 'Скачать JSON',
    }
};

function getLang() {
    return localStorage.getItem('edgebench_lang') || 'ru';
}

function setLang(lang) {
    localStorage.setItem('edgebench_lang', lang);
    location.reload();
}

function t(text) {
    const lang = getLang();
    if (lang === 'en') return text;
    return translations.ru[text] || text;
}

function translatePage() {
    const lang = getLang();

    if (lang === 'ru') {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (translations.ru[key]) el.textContent = translations.ru[key];
        });

        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            if (translations.ru[key]) el.placeholder = translations.ru[key];
        });

        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        while (walker.nextNode()) {
            const node = walker.currentNode;
            const text = node.textContent.trim();
            if (text && translations.ru[text]) {
                node.textContent = node.textContent.replace(text, translations.ru[text]);
            }
        }
    } else {
        // EN mode: restore English text using data-i18n keys + reverse map for bare text nodes
        document.querySelectorAll('[data-i18n]').forEach(el => {
            el.textContent = el.getAttribute('data-i18n');
        });

        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            el.placeholder = el.getAttribute('data-i18n-placeholder');
        });

        // Build reverse map: RU value → EN key (first mapping wins)
        const ruToEn = {};
        for (const [enKey, ruVal] of Object.entries(translations.ru)) {
            if (!ruToEn[ruVal]) ruToEn[ruVal] = enKey;
        }

        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        while (walker.nextNode()) {
            const node = walker.currentNode;
            const text = node.textContent.trim();
            if (text && ruToEn[text]) {
                node.textContent = node.textContent.replace(text, ruToEn[text]);
            }
        }
    }
}

document.addEventListener('DOMContentLoaded', translatePage);
