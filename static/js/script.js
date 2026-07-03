/* ==============================================================================
   ApexTrust - Form Interactivity & Validation Client Script
   ============================================================================== */

document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('.needs-validation');
    const submitBtn = document.querySelector('button[type="submit"]');
    const resetBtn = document.querySelector('button[type="reset"]');

    if (!form) return;

    // 1. Intercept Submit Event and Run Validations
    form.addEventListener('submit', function (event) {
        // Prevent default submission behavior temporarily for evaluation checks
        if (!form.checkValidity() || !runCustomValidations()) {
            event.preventDefault();
            event.stopPropagation();
            form.classList.add('was-validated');
            focusFirstInvalidInput();
        } else {
            // Form is valid: Show Loading Spinner on the predict button
            showLoadingSpinner();
        }
    }, false);

    // 2. Intercept Reset Trigger and Clear Validation Markers
    resetBtn.addEventListener('click', function () {
        form.classList.remove('was-validated');
        
        // Remove manual invalid indicators from custom checks
        const customInvalidFields = form.querySelectorAll('.is-invalid');
        customInvalidFields.forEach(function (field) {
            field.classList.remove('is-invalid');
        });
        
        console.log("Form entries and validation states reset successfully.");
    });

    // 3. Custom Constraints (Numeric & Dropdown Range Validations)
    function runCustomValidations() {
        let isValid = true;

        // FICO Credit Score Validation (Standard 300 to 850)
        const creditScoreInput = document.getElementById('credit_score');
        if (creditScoreInput) {
            const score = parseInt(creditScoreInput.value);
            if (isNaN(score) || score < 300 || score > 850) {
                markFieldInvalid(creditScoreInput, "FICO score must be between 300 and 850.");
                isValid = false;
            } else {
                markFieldValid(creditScoreInput);
            }
        }

        // Age Validation (Min 18 Years)
        const ageInput = document.getElementById('age');
        if (ageInput) {
            const age = parseInt(ageInput.value);
            if (isNaN(age) || age < 18 || age > 100) {
                markFieldInvalid(ageInput, "Applicant must be between 18 and 100 years old.");
                isValid = false;
            } else {
                markFieldValid(ageInput);
            }
        }

        // Debt-to-Income (DTI) Ratio (Between 0.0 and 1.0)
        const dtiInput = document.getElementById('debt_to_income');
        if (dtiInput) {
            const dti = parseFloat(dtiInput.value);
            if (isNaN(dti) || dti < 0.0 || dti > 1.0) {
                markFieldInvalid(dtiInput, "DTI ratio must be a decimal between 0.0 and 1.0.");
                isValid = false;
            } else {
                markFieldValid(dtiInput);
            }
        }

        // Annual Income Minimum Threshold (Min $5,000)
        const incomeInput = document.getElementById('annual_income');
        if (incomeInput) {
            const income = parseFloat(incomeInput.value);
            if (isNaN(income) || income < 5000) {
                markFieldInvalid(incomeInput, "Annual income must be at least $5,000.");
                isValid = false;
            } else {
                markFieldValid(incomeInput);
            }
        }

        // Family Members check (Must be greater than or equal to children + 1)
        const childrenInput = document.getElementById('children_count');
        const familyInput = document.getElementById('family_members');
        if (childrenInput && familyInput) {
            const children = parseInt(childrenInput.value) || 0;
            const family = parseInt(familyInput.value) || 1;
            if (family < (children + 1)) {
                markFieldInvalid(familyInput, "Family members must include applicant plus children count.");
                isValid = false;
            } else {
                markFieldValid(familyInput);
            }
        }

        // Validate all dropdown selections (ensure a non-placeholder is chosen)
        const dropdowns = form.querySelectorAll('select');
        dropdowns.forEach(function (dropdown) {
            if (dropdown.value === "" || dropdown.value === null) {
                markFieldInvalid(dropdown, "Please select an option.");
                isValid = false;
            } else {
                markFieldValid(dropdown);
            }
        });

        return isValid;
    }

    // 4. Helper to Focus the First Erroneous Field
    function focusFirstInvalidInput() {
        const firstInvalid = form.querySelector('.is-invalid, :invalid');
        if (firstInvalid) {
            firstInvalid.focus();
        }
    }

    // 5. Visual Error Indicators Helpers
    function markFieldInvalid(element, errorMessage) {
        element.classList.add('is-invalid');
        
        // Find or create feedback element
        let feedback = element.nextElementSibling;
        if (!feedback || !feedback.classList.contains('invalid-feedback')) {
            feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            element.parentNode.insertBefore(feedback, element.nextSibling);
        }
        feedback.innerText = errorMessage;
    }

    function markFieldValid(element) {
        element.classList.remove('is-invalid');
    }

    // 6. Transition Button to Loading Spinner
    function showLoadingSpinner() {
        // Disable both buttons to prevent multi-clicks during server calculation
        submitBtn.disabled = true;
        resetBtn.disabled = true;

        // Change inner HTML to display Bootstrap spinner
        submitBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Evaluating Profile...
        `;
    }
});
