# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HTML generation module for InspectModule visualization."""

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aitune.torch.jit.inspect_module import InspectModule


class HTMLGenerator:
    """HTML generator for InspectModule hierarchy visualization."""

    @staticmethod
    def generate_html_content(inspect_module_class: type["InspectModule"], model_name: str = "Model") -> str:
        """Generate comprehensive HTML content for module hierarchy visualization.

        Args:
            inspect_module_class: The InspectModule class to generate HTML for.
            model_name: The name of the model to display in the subtitle.

        Returns:
            Complete HTML document as a string with CSS and JavaScript.
        """
        css_styles = HTMLGenerator._get_css_styles()
        javascript = HTMLGenerator._get_javascript()
        module_tree_html = HTMLGenerator._generate_module_tree_html(inspect_module_class)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{html.escape(model_name)} - AITune Model Inspector</title>
            {css_styles}
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 AITune Model Inspector</h1>
                    <p>Model: {html.escape(model_name)}</p>
                </div>
                <div class="content">
                    {module_tree_html}
                </div>
            </div>
            {javascript}
        </body>
        </html>
        """
        return html_content

    @staticmethod
    def _get_css_styles() -> str:
        """Get CSS styles for the HTML page.

        Returns:
            CSS styles as a string.
        """
        return """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #76b900 0%, #4a7c59 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #1a4d1a 0%, #2d5a2d 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 300;
            }
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            .content {
                padding: 30px;
            }
            .module-tree {
                font-family: 'Courier New', monospace;
            }
            .module-item {
                margin: 10px 0;
                border: 1px solid #e1e8ed;
                border-radius: 8px;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            .module-item:hover {
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transform: translateY(-2px);
            }
            .module-header {
                background: linear-gradient(135deg, #76b900 0%, #5a9a00 100%);
                color: white;
                padding: 15px 20px;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                transition: background 0.3s ease;
            }
            .module-header:hover {
                background: linear-gradient(135deg, #5a9a00 0%, #4a7c00 100%);
            }
            .module-name {
                font-weight: bold;
                font-size: 1.1em;
                margin-bottom: 5px;
            }
            .module-full-name {
                font-size: 0.85em;
                opacity: 0.8;
                font-style: italic;
                word-break: break-all;
                margin-bottom: 5px;
            }
            .module-summary {
                font-size: 0.9em;
                opacity: 0.9;
            }
            .search-container {
                margin-bottom: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e1e8ed;
            }
            .search-input {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #76b900;
                border-radius: 6px;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            .search-input:focus {
                outline: none;
                border-color: #5a9a00;
                box-shadow: 0 0 0 3px rgba(118, 185, 0, 0.1);
            }
            .search-results {
                margin-top: 10px;
                font-size: 0.9em;
                color: #7f8c8d;
            }
            .highlighted {
                background-color: #fff3cd;
                border: 2px solid #ffc107;
                animation: highlight 2s ease-in-out;
            }
            @keyframes highlight {
                0% { background-color: #fff3cd; }
                50% { background-color: #ffeaa7; }
                100% { background-color: #fff3cd; }
            }
            .toggle-icon {
                font-size: 1.2em;
                transition: transform 0.3s ease;
            }
            .toggle-icon.expanded {
                transform: rotate(90deg);
            }
            .module-details {
                background: #f8f9fa;
                padding: 20px;
                display: none;
                border-top: 1px solid #e1e8ed;
            }
            .module-details.expanded {
                display: block;
                animation: slideDown 0.3s ease;
            }
            @keyframes slideDown {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            .detail-section {
                margin: 15px 0;
                padding: 15px;
                background: white;
                border-radius: 6px;
                border-left: 4px solid #76b900;
            }
            .detail-section h4 {
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            .detail-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 10px;
            }
            .detail-item {
                background: #ecf0f1;
                padding: 10px;
                border-radius: 4px;
                border-left: 3px solid #76b900;
            }
            .detail-label {
                font-weight: bold;
                color: #2c3e50;
                font-size: 0.9em;
            }
            .detail-value {
                color: #34495e;
                margin-top: 5px;
                word-break: break-all;
            }
            .args-list, .kwargs-list {
                background: #e8f5e8;
                padding: 5px 10px;
                border-radius: 4px;
                margin-top: 5px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                white-space: pre-line;
            }
            .no-modules {
                text-align: center;
                padding: 60px 20px;
                color: #7f8c8d;
                font-size: 1.2em;
            }
            .no-modules-icon {
                font-size: 4em;
                margin-bottom: 20px;
                opacity: 0.5;
            }
            .stats-bar {
                background: linear-gradient(135deg, #76b900 0%, #5a9a00 100%);
                color: white;
                padding: 15px 20px;
                margin-bottom: 20px;
                border-radius: 8px;
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
            }
            .stat-item {
                text-align: center;
                margin: 5px;
            }
            .stat-number {
                font-size: 1.5em;
                font-weight: bold;
                display: block;
            }
            .stat-label {
                font-size: 0.9em;
                opacity: 0.9;
            }
            .toggle-button {
                background: #5a9a00;
                color: white;
                border: none;
                padding: 10px 20px;
                margin: 0 10px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .toggle-button:hover {
                background: #4a7c00;
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .toggle-button.active {
                background: #d32f2f;
                box-shadow: 0 0 0 3px rgba(211, 47, 47, 0.3);
            }
            .toggle-button.active:hover {
                background: #b71c1c;
            }
            .toggle-button::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            .toggle-button:hover::before {
                left: 100%;
            }
            @media (max-width: 768px) {
                .container {
                    margin: 10px;
                    border-radius: 10px;
                }
                .header {
                    padding: 20px;
                }
                .header h1 {
                    font-size: 2em;
                }
                .content {
                    padding: 20px;
                }
                .detail-grid {
                    grid-template-columns: 1fr;
                }
                .stats-bar {
                    flex-direction: column;
                }
            }
        </style>
        """

    @staticmethod
    def _get_javascript() -> str:
        """Get JavaScript for interactive functionality.

        Returns:
            JavaScript code as a string.
        """
        return """
        <script>
            function toggleModule(element) {
                const details = element.nextElementSibling;
                const icon = element.querySelector('.toggle-icon');
                if (details.classList.contains('expanded')) {
                    details.classList.remove('expanded');
                    icon.classList.remove('expanded');
                } else {
                    details.classList.add('expanded');
                    icon.classList.add('expanded');
                }
            }
            let isExpandedState = false; // Track the current expand/collapse state
            let expandCollapseButton = null; // Reference to the expand/collapse toggle button

            function toggleExpandCollapse() {
                const headers = document.querySelectorAll('.module-header');
                const details = document.querySelectorAll('.module-details');
                const icons = document.querySelectorAll('.toggle-icon');

                // Toggle the state
                isExpandedState = !isExpandedState;

                // Update element visibility based on new state
                headers.forEach((header, index) => {
                    if (isExpandedState) {
                        details[index].classList.add('expanded');
                        icons[index].classList.add('expanded');
                    } else {
                        details[index].classList.remove('expanded');
                        icons[index].classList.remove('expanded');
                    }
                });

                // Update button text and visual state using the stored reference
                if (expandCollapseButton) {
                    expandCollapseButton.textContent = isExpandedState ? 'Collapse All' : 'Expand All';
                    // Toggle active class for visual feedback
                    if (isExpandedState) {
                        expandCollapseButton.classList.add('active');
                    } else {
                        expandCollapseButton.classList.remove('active');
                    }
                }
            }
            function searchModules() {
                const searchTerm = document.getElementById('moduleSearch').value.toLowerCase().trim();
                const modules = document.querySelectorAll('.module-item');
                const resultsDiv = document.getElementById('searchResults');
                let foundCount = 0;

                // Remove previous highlights
                modules.forEach(module => {
                    module.classList.remove('highlighted');
                });

                if (searchTerm === '') {
                    resultsDiv.textContent = '';
                    return;
                }

                modules.forEach(module => {
                    const moduleNameEl = module.querySelector('.module-name');
                    const fullNameEl = module.querySelector('.module-full-name');

                    // Check if elements exist before accessing their textContent
                    if (!moduleNameEl || !fullNameEl) {
                        return;
                    }

                    const moduleName = moduleNameEl.textContent.toLowerCase();
                    const fullName = fullNameEl.textContent.toLowerCase();

                    if (moduleName.includes(searchTerm) || fullName.includes(searchTerm)) {
                        module.classList.add('highlighted');
                        foundCount++;

                        // Expand the module and scroll to it
                        const header = module.querySelector('.module-header');
                        const details = module.querySelector('.module-details');
                        const icon = module.querySelector('.toggle-icon');

                        if (details && icon && !details.classList.contains('expanded')) {
                            details.classList.add('expanded');
                            icon.classList.add('expanded');
                        }

                        // Scroll to the first found module
                        if (foundCount === 1) {
                            module.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        }
                    }
                });

                resultsDiv.textContent = foundCount > 0 ?
                    `Found ${foundCount} module(s) matching "${searchTerm}"` :
                    `No modules found matching "${searchTerm}"`;
            }

            function handleSearchKeyPress(event) {
                if (event.key === 'Enter') {
                    searchModules();
                }
            }
            function clearSearch() {
                document.getElementById('moduleSearch').value = '';
                document.getElementById('searchResults').textContent = '';
                document.querySelectorAll('.module-item').forEach(module => {
                    module.classList.remove('highlighted');
                });
            }
            let isDetailedMode = false; // Track the current state
            let toggleButton = null; // Reference to the toggle button

            function toggleDetailedTypes() {
                const typeElements = document.querySelectorAll('.type-display');

                // Toggle the state
                isDetailedMode = !isDetailedMode;

                // Update each type display element
                typeElements.forEach(el => {
                    if (isDetailedMode) {
                        // Show full type names - restore original content
                        el.innerHTML = el.getAttribute('data-full-content') || el.innerHTML;
                    } else {
                        // Show short type names - strip package names
                        const fullContent = el.getAttribute('data-full-content') || el.innerHTML;
                        if (!el.getAttribute('data-full-content')) {
                            el.setAttribute('data-full-content', fullContent); // Store original for restoration
                        }

                        // Strip package names from type strings
                        const shortContent = fullContent.replace(/(\\w+\\.)+(\\w+)/g, '$2');
                        el.innerHTML = shortContent;
                    }
                });

                // Update button text and visual state using the stored reference
                if (toggleButton) {
                    toggleButton.textContent = isDetailedMode ? 'Show Short Types' : 'Show Detailed Types';
                    // Toggle active class for visual feedback
                    if (isDetailedMode) {
                        toggleButton.classList.add('active');
                    } else {
                        toggleButton.classList.remove('active');
                    }
                }
            }

            function initializeTypeDisplay() {
                // Initialize all type displays to show short versions
                const typeElements = document.querySelectorAll('.type-display');
                typeElements.forEach(el => {
                    const fullContent = el.innerHTML;
                    el.setAttribute('data-full-content', fullContent); // Store original for restoration

                    // Strip package names from type strings to show short version initially
                    const shortContent = fullContent.replace(/(\\w+\\.)+(\\w+)/g, '$2');
                    el.innerHTML = shortContent;
                });
            }
            document.addEventListener('DOMContentLoaded', function() {
                const content = document.querySelector('.content');

                // Create search container
                const searchContainer = document.createElement('div');
                searchContainer.className = 'search-container';
                searchContainer.innerHTML = `
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <input type="text" id="moduleSearch" class="search-input" placeholder="Search modules by name..." onkeypress="handleSearchKeyPress(event)">
                        <button onclick="searchModules()" style="background: #76b900; color: white; border: none; padding: 12px 15px; border-radius: 6px; cursor: pointer; font-size: 14px; transition: background 0.3s ease;" onmouseover="this.style.background='#5a9a00'" onmouseout="this.style.background='#76b900'">Search</button>
                        <button onclick="clearSearch()" style="background: #95a5a6; color: white; border: none; padding: 12px 15px; border-radius: 6px; cursor: pointer; font-size: 14px; transition: background 0.3s ease;" onmouseover="this.style.background='#7f8c8d'" onmouseout="this.style.background='#95a5a6'">Clear</button>
                    </div>
                    <div id="searchResults" class="search-results"></div>
                `;

                // Create button container
                const buttonContainer = document.createElement('div');
                buttonContainer.style.cssText = 'margin-bottom: 20px; text-align: center;';
                const expandCollapseBtn = document.createElement('button');
                expandCollapseBtn.textContent = 'Expand All';
                expandCollapseBtn.className = 'toggle-button';
                expandCollapseBtn.onclick = toggleExpandCollapse;
                expandCollapseButton = expandCollapseBtn; // Store reference for the toggle function
                const toggleTypesBtn = document.createElement('button');
                toggleTypesBtn.textContent = 'Show Detailed Types';
                toggleTypesBtn.className = 'toggle-button';
                toggleTypesBtn.onclick = toggleDetailedTypes;
                toggleButton = toggleTypesBtn; // Store reference for the toggle function
                buttonContainer.appendChild(expandCollapseBtn);
                buttonContainer.appendChild(toggleTypesBtn);

                // Insert search container and button container
                content.insertBefore(searchContainer, content.firstChild);
                content.insertBefore(buttonContainer, content.firstChild);

                // Initialize type displays to show short versions
                initializeTypeDisplay();
            });
        </script>
        """

    @staticmethod
    def _generate_module_tree_html(inspect_module_class: type["InspectModule"]) -> str:
        """Generate HTML for the module tree structure.

        Args:
            inspect_module_class: The InspectModule class to generate HTML for.

        Returns:
            HTML string representing the module hierarchy.
        """
        if not inspect_module_class.heads:
            return """
            <div class="no-modules">
                <div class="no-modules-icon">📭</div>
                <div>No modules found in hierarchy</div>
            </div>
            """

        def _generate_module_html(module: "InspectModule", level: int = 0) -> str:
            """Generate HTML for a single module and its children.

            Args:
                module: The module to generate HTML for.
                level: The current hierarchy level.

            Returns:
                HTML string for the module and its children.
            """
            # Generate unique ID for this module
            module_id = f"module_{id(module)}"
            # Format execution time
            exec_time = f"{module._total_execution_time:.3f}s" if module._total_execution_time > 0 else "0.000s"
            # Format args with separate lines for each item - store full type names, JS will handle short/long display
            args_html = ""
            if module._args:
                args_items = [f"  {i}: {arg}" for i, arg in enumerate(module._args)]
                args_html += '<div class="args-list type-display">' + "\n".join(args_items) + "</div>"
            # Format kwargs with separate lines for each item - store full type names, JS will handle short/long display
            kwargs_html = ""
            if module._kwargs:
                kwargs_items = [f"  {k}: {v}" for k, v in module._kwargs.items()]
                kwargs_html += '<div class="kwargs-list type-display">' + "\n".join(kwargs_items) + "</div>"
            # Format output types with separate lines for each item - store full type names, JS will handle short/long display
            output_types_html = ""
            if module._output_types is not None:
                if isinstance(module._output_types, (list, tuple)):
                    output_items = [f"  {i}: {item}" for i, item in enumerate(module._output_types)]
                elif isinstance(module._output_types, dict):
                    output_items = [f"  {k}: {v}" for k, v in module._output_types.items()]
                else:
                    output_items = [f"  {module._output_types}"]
                output_types_html += '<div class="args-list type-display">' + "\n".join(output_items) + "</div>"
            # Generate children HTML
            children_html = ""
            for child in module._children:
                children_html += _generate_module_html(child, level + 1)

            return f"""
            <div class="module-item">
                <div class="module-header" onclick="toggleModule(this)">
                    <div>
                        <div class="module-name">{module._name}</div>
                        <div class="module-full-name">{module._name_detailed}</div>
                        <div class="module-summary">Level {module._level} • {module._call_count} calls • Total time {exec_time} • Self time: {HTMLGenerator._calculate_self_time(module):.3f}s</div>
                    </div>
                    <div class="toggle-icon">▶</div>
                </div>
                <div class="module-details" id="{module_id}">
                    <div class="detail-section">
                        <h4>📋 Basic Information</h4>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Class Name</div>
                                <div class="detail-value">{module._name.split(" 📊")[0]}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Hierarchy Level</div>
                                <div class="detail-value">{module._level}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Call Count</div>
                                <div class="detail-value">{module._call_count}</div>
                            </div>
                        </div>
                    </div>
                    <div class="detail-section">
                        <h4>⚡ Execution Details</h4>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Total Execution Time</div>
                                <div class="detail-value">{exec_time}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Self Time</div>
                                <div class="detail-value">{HTMLGenerator._calculate_self_time(module):.3f}s</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Average Time per Call</div>
                                <div class="detail-value">{exec_time if module._call_count == 0 else f"{module._total_execution_time / module._call_count:.3f}s"}</div>
                            </div>
                        </div>
                    </div>
                    <div class="detail-section">
                        <h4>📥 Input Arguments</h4>
                        <div class="detail-item">
                            <div class="detail-label">Args Types</div>
                            {args_html if args_html else '<div class="detail-value">No arguments</div>'}
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Kwargs Types</div>
                            {kwargs_html if kwargs_html else '<div class="detail-value">No keyword arguments</div>'}
                        </div>
                    </div>
                    <div class="detail-section">
                        <h4>📤 Output Information</h4>
                        <div class="detail-item">
                            <div class="detail-label">Result Types</div>
                            {output_types_html if output_types_html else '<div class="detail-value">No output data</div>'}
                        </div>
                    </div>
                    {f'<div class="detail-section"><h4>🌳 Child Modules ({len(module._children)})</h4>{children_html}</div>' if module._children else ""}
                </div>
            </div>
            """

        # Generate HTML for all top-level modules
        modules_html = ""
        for head in inspect_module_class.heads:
            modules_html += _generate_module_html(head, 0)
        return f'<div class="module-tree">{modules_html}</div>'

    @staticmethod
    def _traverse_modules(inspect_module_class: type["InspectModule"]):
        """Generator that traverses all modules in the hierarchy.

        Args:
            inspect_module_class: The InspectModule class to traverse.

        Yields:
            All InspectModule instances in the hierarchy.
        """

        def _traverse(module: "InspectModule"):
            yield module
            for child in module._children:
                yield from _traverse(child)

        for head in inspect_module_class.heads:
            yield from _traverse(head)

    @staticmethod
    def _calculate_self_time(module: "InspectModule") -> float:
        """Calculate self time for a module.

        Self time is the total execution time minus the sum of all children's execution times.

        Args:
            module: The module to calculate self time for.

        Returns:
            Self time in seconds.
        """
        children_total_time = sum(child._total_execution_time for child in module._children)
        return max(0.0, module._total_execution_time - children_total_time)
