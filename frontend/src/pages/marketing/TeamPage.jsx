import React from 'react';
import MarketingLayout from './MarketingLayout';

const TeamPage = () => (
  <MarketingLayout>
    <div className="bloc tc-502 bgc-4279 l-bloc" id="team">
      <div className="container bloc-md-sm bloc-md bloc-lg-md">
        <div className="row">
          <div className="text-center col-lg-6 offset-lg-3">
            <h3 className="page-title tc-458 mb-4">Наша команда</h3>
            <p className="sub-heading mb-4">
              Мы создаем инновационные решения для автоматизированной торговли, сочетая передовые алгоритмы машинного обучения с глубоким пониманием рыночных механизмов.
            </p>
          </div>
        </div>
        <div className="row voffset">
          {[
            { img: 'Egor', role: 'Egor - CEO', desc: 'Стратегическое видение, общее руководство' },
            { img: 'John', role: 'John - CTO', desc: 'Архитектура платформы, R&D' },
            { img: 'ZhangLi', role: 'Zhang Li - Lead Data Scientist', desc: 'Разработка и обучение торговых моделей ИИ' },
            { img: 'Paul', role: 'Paul - Senior Algorithm Developer', desc: 'Создание и оптимизация торговых стратегий' },
            { img: 'Alex', role: 'Alex - Quantitative Researcher', desc: 'Математическое моделирование стратегий' },
            { img: 'Helen', role: 'Helen - Financial Markets Analyst', desc: 'Анализ рыночных тенденций и рисков' },
          ].map(member => (
            <div className="col-lg-3 col-md-6 text-center" key={member.role}>
              <img src={`/img/${member.img}.jpg`} className="img-fluid mx-auto d-block" alt={member.role} width="267" height="267" />
              <h3 className="tc-458 mt-3">{member.role}</h3>
              <p>{member.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  </MarketingLayout>
);

export default TeamPage;



