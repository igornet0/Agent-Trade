import React from 'react';
import MarketingLayout from './MarketingLayout';

const ContactPage = () => (
  <MarketingLayout>
    <div className="bloc bgc-4279 l-bloc" id="contact">
      <div className="container bloc-xl-lg bloc-md bloc-lg-md">
        <div className="row">
          <div className="text-md-start col-lg-8 offset-lg-2 text-sm-start text-start offset-md-1 col-sm-12 offset-sm-0 col-md-10">
            <h2 className="text-start page-title tc-458 mb-4">Contact</h2>
            <p className="text-start sub-heading mb-4">
              Please feel free to get in touch with any of your questions regarding our product and service. We will respond as soon as possible.
            </p>
            <div className="divider-h primary-gradient-bg site-divider"></div>
            <form id="contact_form" data-form-type="blocs-form" noValidate data-success-msg="Your message has been sent." data-fail-msg="Sorry it seems that our mail server is not responding, Sorry for the inconvenience!">
              <div className="form-group mb-3">
                <label className="form-label">Name</label>
                <input id="name" className="form-control" required />
              </div>
              <div className="form-group mb-3">
                <label className="form-label">Email</label>
                <input id="email" className="form-control" type="email" required />
              </div>
              <div className="form-group mb-3">
                <label className="form-label">Message</label>
                <textarea id="message" className="form-control" rows="4" cols="50" required />
              </div>
              <div className="form-check">
                <input className="form-check-input" type="checkbox" id="optin" name="optin" />
                <label className="form-check-label">
                  We can send you product updates and offers via email. It is possible to opt-out at any time.
                </label>
              </div>
              <button className="bloc-button btn btn-lg form-btn btn-c-7096 mt-3" type="submit">Send Message</button>
            </form>
          </div>
        </div>
      </div>
    </div>
  </MarketingLayout>
);

export default ContactPage;




