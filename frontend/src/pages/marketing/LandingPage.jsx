import React from 'react';
import MarketingLayout from './MarketingLayout';

const LandingPage = () => {
  return (
    <MarketingLayout>
      {/* hero */}
      <div className="bloc bloc-fill-screen primary-gradient-bg bgc-4580 l-bloc" id="hero">
        <div className="container fill-bloc-top-edge">
          <div className="row">
            <div className="col-12">
              {/* navbar lives in layout */}
            </div>
          </div>
        </div>
        <div className="container">
          <div className="row">
            <div className="offset-lg-1 col-lg-10">
              <h1 className="hero-heading tc-4279 mb-4 text-center">
                <span>Торговля с помощью ИИ</span>
              </h1>
              <h3 className="float-none text-center hero-sub-heading h3-style tc-4279">
                Доверьте свои инвестиции нейросетям нового поколения. Наша система анализирует рынок 24/7 и совершает сделки с максимальной эффективностью.
              </h3>
              <div className="text-center">
                <a href="/profile_page" className="btn btn-rd btn-lg btn-d mt-lg-2 mb-lg-2 pt-lg-3 pb-lg-3">Начать сейчас</a>
              </div>
            </div>
          </div>
        </div>
        <div className="container fill-bloc-bottom-edge">
          <div className="row">
            <div className="col-12 col-lg-10 offset-lg-1">
              <div className="text-center">
                <img src="/img/app-screenshot-short.jpg" className="img-fluid mx-auto d-block app-offset-bottom hero-img" alt="app screenshot-short" width="946" height="552" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* feature-cards */}
      <div className="bloc bgc-4279 l-bloc animSpeedFast" id="feature-cards">
        <div className="container bloc-lg bloc-xl-lg bloc-lg-sm">
          <div className="row nonein mt-lg-0">
            <div className="col-md-6 col-sm-6 col-lg-5 text-lg-start align-self-end offset-lg-1">
              <div className="card border-0 feature-card bgc-3202">
                <div className="card-body text-center">
                  <img src="/img/swap-svgrepo-com.svg" className="img-fluid mx-auto d-block feature-icon feature-icon-center img-frame-md" alt="Автоматическая торговля" width="22" height="22" />
                  <h4 className="text-center tc-458 mt-3 mb-3 btn-resize-mode h4-style">Автоматическая торговля</h4>
                  <p className="text-center">Система самостоятельно совершает сделки на основе прогнозов нейросети</p>
                </div>
              </div>
            </div>
            <div className="col-md-6 col-sm-6 col-lg-5">
              <div className="card border-0 feature-card bgc-3202">
                <div className="card-body text-center">
                  <img src="/img/analysis-analytics-data-svgrepo-com-2.svg" className="img-fluid mx-auto d-block feature-icon feature-icon-center" alt="Глубокий анализ" width="40" height="40" />
                  <h4 className="text-center tc-458 mt-3 mb-3 h4-feature-cards-style">Глубокий анализ</h4>
                  <p className="text-center">Искусственный интеллект анализирует тысячи рыночных показателей</p>
                </div>
              </div>
            </div>
            <div className="col-md-6 col-sm-6 col-lg-5 offset-lg-1">
              <div className="card bgc-3202 border-0 feature-card">
                <div className="card-body text-center">
                  <img src="/img/captcha-automatic-code-svgrepo-com.svg" className="img-fluid feature-icon feature-icon-center mx-auto img-frame" alt="Безопасность" width="30" height="30" />
                  <h4 className="text-center tc-458 mt-3 mb-3 h4-feature-cards-style">Безопасность</h4>
                  <p className="text-center">Ваши средства защищены многоуровневой системой безопасности</p>
                </div>
              </div>
            </div>
            <div className="col-md-6 col-sm-6 col-lg-5">
              <div className="card feature-card border-0 bgc-3202">
                <div className="card-body text-center">
                  <img src="/img/support-svgrepo-com-2.svg" className="img-fluid mx-auto d-block feature-icon feature-icon-center" alt="Поддержка 24/7" width="40" height="40" />
                  <h4 className="text-center tc-458 mt-3 mb-3 h4-feature-cards-style">Поддержка 24/7</h4>
                  <p className="text-center">Круглосуточный мониторинг и техническая поддержка</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* multiple-features */}
      <div className="bloc tc-426 primary-gradient-bg l-bloc" id="multiple-features">
        <div className="container bloc-md-sm bloc-lg-md bloc-lg-lg bloc-md">
          <div className="row">
            <div className="col-lg-12 col-md-12 text-center text-sm-start">
              <h2 className="feature-heading tc-4279 mb-4">Как это работает?</h2>
              <p className="sub-heading tc-426 mb-4">
                Наша нейросетевая платформа использует комбинацию глубинного обучения и анализа больших данных для прогнозирования рынка. Каждый этап процесса оптимизирован для максимизации прибыли при минимальных рисках.
              </p>
            </div>
            {[
              { title: '1. Ввод средств', text: 'Подключите криптовалютный кошелек или воспользуйтесь банковской картой для депозита. Минимальная сумма инвестиций - $50. Все транзакции защищены SSL-шифрованием.' },
              { title: '2. Анализ рынка', text: 'Наша LSTM-нейросеть анализирует в реальном времени: Исторические данные, Объемы торгов, Новостной фон, Социальные тренды, Цепные реакции рынка.' },
              { title: '3. Автоторговля', text: 'Система совершает сделки через API-интеграции с топовыми биржами (Binance, Coinbase, OKX). Используются стратегии: Арбитраж, Скальпинг, Трендовая торговля, Mean reversion.' },
              { title: '4. Вывод прибыли', text: 'Выводите средства в любой момент через удобный интерфейс. Комиссия на вывод - 0.5%. Ежедневная отчетность и детальная аналитика всех сделок в личном кабинете.' },
            ].map((item) => (
              <div className="col-lg-3 col-md-6 col-sm-6 text-center text-sm-start feature-col" key={item.title}>
                <h4 className="tc-4279 mt-3 mb-3">{item.title}</h4>
                <p className="tc-426">{item.text}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* call-to */}
      <div className="bloc tc-426 none bgc-4279 l-bloc" id="call-to">
        <div className="container bloc-xl-lg bloc-md-sm bloc-lg-md bloc-md">
          <div className="row">
            <div className="col">
              <div className="cta-card primary-gradient-bg custom-shadow">
                <div className="row g-0">
                  <div className="col-lg-5 col-md-12 text-center text-lg-start feature-col">
                    <h2 className="feature-heading tc-4279 mt-md-0 mt-lg-0 mt-3 mb-4 h2-style">Мониторинг и управление инвестициями</h2>
                    <p className="text-center float-lg-none text-lg-start sub-heading tc-426 text-md-start p-10-style">В личном кабинете доступен полный контроль над инвестициями:</p>
                    <ul className="list-style">
                      {[
                        'Реальная статистика доходности',
                        'Детализация всех сделок',
                        'Настройка рисковых профилей',
                        'Графики изменения капитала',
                        'Экспорт отчетов в PDF/CSV',
                      ].map((txt) => (
                        <li key={txt}><p className="text-md-start tc-4279">{txt}</p></li>
                      ))}
                    </ul>
                    <a href="/pricing_page" className="btn white-btn btn-lg btn-c-4279 mt-lg-4 ms-lg-5 mb-lg-5">Начать сейчас</a>
                  </div>
                  <div className="col-md-12 offset-lg-0 col-lg-7 d-sm-block d-none">
                    <img src="/img/app-screenshot-hero.jpg" className="img-fluid float-lg-none cta-card-img" alt="App Screenshot" width="595" height="351" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* scroll to top */}
      <button aria-label="Scroll to top button" className="bloc-button btn btn-d scrollToTop" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
        <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 32 32"><path className="scroll-to-top-btn-icon" d="M30,22.656l-14-13-14,13"/></svg>
      </button>
    </MarketingLayout>
  );
};

export default LandingPage;



