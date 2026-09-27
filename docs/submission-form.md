# Ссылки для формы на платформе хакатона

Репозиторий: https://github.com/Fgeeha/mt-hackathon-tram-forecast

1. Артефакты ML-модели, код обучения/инференса, README с инструкцией запуска:
   https://github.com/Fgeeha/mt-hackathon-tram-forecast
   модель и прогнозы: https://github.com/Fgeeha/mt-hackathon-tram-forecast/tree/master/artifacts
   код: https://github.com/Fgeeha/mt-hackathon-tram-forecast/tree/master/src/tram_forecast
   файл прогноза: https://github.com/Fgeeha/mt-hackathon-tram-forecast/blob/master/artifacts/submission_calibrated_v3.csv (WAPE-score 0.89037)
2. Внешние данные (все источники, ссылки и эффект в бэктесте):
   https://github.com/Fgeeha/mt-hackathon-tram-forecast#внешние-источники-и-подтверждение-эффекта
3. Запускаемый веб-сервис (Docker Compose, backend + frontend, API, инструкция для жюри):
   https://github.com/Fgeeha/mt-hackathon-tram-forecast#быстрый-старт-для-жюри
   API: https://github.com/Fgeeha/mt-hackathon-tram-forecast#api
   готовый образ: ghcr.io/fgeeha/mt-hackathon-tram-forecast:latest (публикуется CI: https://github.com/Fgeeha/mt-hackathon-tram-forecast/actions)
4. Схема архитектуры и модулей, область определения и адаптации, зависимости от внешних данных:
   https://github.com/Fgeeha/mt-hackathon-tram-forecast#архитектура
   https://github.com/Fgeeha/mt-hackathon-tram-forecast#область-применимости
5. Производительность (замеры, журнал нагрузочного теста) и дополнительные возможности:
   https://github.com/Fgeeha/mt-hackathon-tram-forecast#производительность
   журнал прогона: https://github.com/Fgeeha/mt-hackathon-tram-forecast/blob/master/docs/loadtest-2026-09-27.log
   https://github.com/Fgeeha/mt-hackathon-tram-forecast#реализованные-дополнительные-возможности
6. Ограничения решения и план развития:
   https://github.com/Fgeeha/mt-hackathon-tram-forecast#ограничения-и-план-развития
