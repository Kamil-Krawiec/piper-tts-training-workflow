// Run with a saved dataset selected and all steps available:
// agent-browser eval --stdin < scripts/check-ui-layout.js
(async () => {
  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
  const visible = selector => [...document.querySelectorAll(selector)].find(element => element.getBoundingClientRect().height > 0);
  const bounds = element => {
    const box = element.getBoundingClientRect();
    return {top: box.top, height: box.height, bottom: box.bottom, left: box.left, width: box.width, right: box.right};
  };
  const measurements = [];
  for (const step of [3, 4, 1, 2, 5, 6, 7, 3]) {
    const tab = document.querySelector(`#workflow [role=tab][data-tab-id="${step}"]`);
    if (!tab || tab.getAttribute('aria-disabled') === 'true') {
      throw new Error('Select a project with a saved dataset to check all seven pages.');
    }
    tab.click();
    await wait(300);
    const page = visible('.step-page');
    measurements.push({step, ...bounds(page), footer: bounds(visible('.step-footer')).bottom});
    if (document.documentElement.scrollWidth > innerWidth || document.documentElement.scrollHeight > innerHeight + 1) {
      throw new Error(`Step ${step} overflows the browser viewport.`);
    }
    const content = page.querySelector('.step-content');
    if (!content || content.clientHeight < 80) throw new Error(`Step ${step} has no usable scrolling area.`);
    content.scrollTop = content.scrollHeight;
    await wait(50);
    if (Math.abs(bounds(page).top - measurements[0].top) > 1) throw new Error('Scrolling moves the workspace.');
  }
  for (const value of measurements) {
    for (const key of ['top', 'height', 'bottom', 'left', 'width', 'right', 'footer']) {
      if (Math.abs(value[key] - measurements[0][key]) > 1) {
        throw new Error(`Step ${value.step} changes the workspace ${key}.`);
      }
    }
  }
  return {viewport: [innerWidth, innerHeight], measurements, result: 'Stable page width, position and footer; no viewport overflow'};
})()
