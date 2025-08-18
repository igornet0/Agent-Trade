import React from 'react';
import MarketingLayout from './MarketingLayout';

const questions = [
  'How do I use your product?',
  'How do I get in touch?',
  'How long is the trial?',
  'How do I get setup?',
  'Can I share my account?',
  'Do you offer a subscription?',
  'Do you offer student discount?',
  'How do I return an item?',
  'Do you offer refunds?'
];

const FaqPage = () => (
  <MarketingLayout>
    <div className="bloc bgc-4279 tc-502 l-bloc" id="faq">
      <div className="container bloc-xl-lg bloc-md-sm bloc-lg-md bloc-md">
        <div className="row">
          <div className="col-lg-12 col-md-12">
            <h2 className="text-start page-title tc-458 mb-4">Frequently asked questions</h2>
            <p className="text-start sub-heading mb-4">
              If you have any other questions you want to ask, please <a className="ltc-7096" href="/contact_page">get in touch</a>.
            </p>
          </div>
          {questions.map(q => (
            <div className="col-lg-4 col-md-6" key={q}>
              <h4 className="text-start tc-458 mb-3">{q}</h4>
              <p className="mg-lg text-start">Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor.</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  </MarketingLayout>
);

export default FaqPage;




