// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export default function CustomFooter() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bd-footer">
      <div className="bd-footer__inner">
        <div className="footer-items__start">
          <div className="footer-item">
            <div className="footer-logos-container">
              <a
                className="footer-brand"
                href="https://www.nvidia.com"
                target="_blank"
                rel="noopener"
              >
                NVIDIA AITune
              </a>
            </div>
          </div>
          <div className="footer-item">
            <div className="footer-links">
              <a href="https://www.nvidia.com/en-us/about-nvidia/privacy-policy/" target="_blank" rel="noopener">Privacy Policy</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/about-nvidia/privacy-center/" target="_blank" rel="noopener">Your Privacy Choices</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/about-nvidia/terms-of-service/" target="_blank" rel="noopener">Terms of Service</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/about-nvidia/accessibility/" target="_blank" rel="noopener">Accessibility</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/contact/" target="_blank" rel="noopener">Contact</a>
            </div>
          </div>
          <div className="footer-item">
            <p className="copyright">Copyright &copy; {currentYear}, NVIDIA Corporation.</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
