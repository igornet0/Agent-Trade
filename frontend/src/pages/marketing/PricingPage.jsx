import React from 'react';
import MarketingLayout from './MarketingLayout';

const plans = [
  { price: '$9', name: 'Starter', light: false },
  { price: '$15', name: 'Small business', light: true },
  { price: '$39', name: 'Enterprise', light: false },
];

const PricingPage = () => (
  <MarketingLayout>
    <div className="bloc bgc-4279 l-bloc" id="download">
      <div className="container bloc-md bloc-lg-md bloc-lg-lg">
        <div className="row">
          <div className="text-center text-md-start col-lg-8 offset-lg-2 text-lg-start">
            <h2 className="page-title text-center tc-458 mb-4">Pricing</h2>
            <p className="sub-heading text-center mb-4">We offer three different packages so you can pick the one that’s right for you. It doesn’t matter what size your company is, we got you covered.</p>
          </div>
          {plans.map((p) => (
            <div className="text-center text-md-start offset-lg-0 col-lg-4 align-self-center col" key={p.name}>
              <div className={`price-card text-lg-start ${p.light ? 'primary-gradient-bg' : ''}`}>
                <p className={`text-start page-title ${p.light ? 'tc-4279' : 'tc-458'} mb-4`}>{p.price}</p>
                <h3 className={`text-start ${p.light ? 'tc-4279' : 'tc-458'} mb-3`}>{p.name}</h3>
                <p className={`text-start pricing-info ${p.light ? 'tc-426' : ''}`}>Some Feature Information</p>
                <a href="/" className={`btn btn-lg w-100 ${p.light ? 'white-btn btn-c-4279' : 'btn-c-7096'}`}>Get Started</a>
                <ul className="list-unstyled list-sp-md mt-5">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <li key={i} className={i === 0 && p.name === 'Enterprise' ? 'pt-5' : ''}>
                      <img src={`/img/${p.light ? 'tick-icon-light.svg' : 'tick-icon.svg'}`} className={`img-fluid float-lg-none price-list-icon ${p.light ? '' : 'dark-icon'}`} alt="tick" width="20" height="20" />
                      <p className={`price-list-item ${p.light ? 'tc-4279' : ''}`}>Some Feature Information</p>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  </MarketingLayout>
);

export default PricingPage;




