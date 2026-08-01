/* ============================================================
   MelonVision CNN - Main JavaScript
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

  // ── Sidebar Toggle ──────────────────────────────────────────
  const sidebar       = document.getElementById('sidebar');
  const mainWrapper   = document.getElementById('main-wrapper');
  const toggleBtn     = document.getElementById('sidebarToggle');

  // Create overlay for mobile
  const overlay = document.createElement('div');
  overlay.className = 'sidebar-overlay';
  document.body.appendChild(overlay);

  const isMobile = () => window.innerWidth < 992;

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      if (isMobile()) {
        sidebar.classList.toggle('open');
        overlay.classList.toggle('show');
      } else {
        document.body.classList.toggle('sidebar-collapsed');
      }
    });
  }

  overlay.addEventListener('click', () => {
    sidebar.classList.remove('open');
    overlay.classList.remove('show');
  });

  // Close sidebar on resize back to desktop
  window.addEventListener('resize', () => {
    if (!isMobile()) {
      sidebar.classList.remove('open');
      overlay.classList.remove('show');
    }
  });

  // ── Topbar Date ─────────────────────────────────────────────
  const dateEl = document.getElementById('topbar-date');
  if (dateEl) {
    const now = new Date();
    dateEl.textContent = now.toLocaleDateString('id-ID', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
    });
  }

  // ── Auto-dismiss alerts ─────────────────────────────────────
  document.querySelectorAll('.alert').forEach(alert => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      if (bsAlert) bsAlert.close();
    }, 5000);
  });

  // ── Tooltips ────────────────────────────────────────────────
  document.querySelectorAll('[title]').forEach(el => {
    new bootstrap.Tooltip(el, { trigger: 'hover', placement: 'top' });
  });

});
