/* ==============================================================================
   ApexTrust - Client-Side Machine Learning Prediction & UI Interactivity Script
   ============================================================================== */

// Model weights, intercept, and scaling parameters extracted from the trained model pipeline
const scaler_params = {
    children_count: { mean: 0.435833333, scale: 0.755567759 },
    annual_income: { mean: 52365.8125, scale: 19574.8841 },
    age: { mean: 45.5858333, scale: 16.5327745 },
    years_employed: { mean: 10.1945, scale: 9.23111106 },
    work_phone: { mean: 0.238333333, scale: 0.426064028 },
    email: { mean: 0.176666667, scale: 0.381386360 },
    family_members: { mean: 2.25916667, scale: 0.837853988 },
    credit_score: { mean: 623.558333, scale: 127.920854 },
    existing_loans: { mean: 2.57083333, scale: 1.6683273 },
    debt_to_income: { mean: 0.224232078, scale: 0.125759373 },
    income_per_family_member: { mean: 26611.9293, scale: 15490.2588 },
    employment_to_age_ratio: { mean: 0.20034882, scale: 0.144661566 },
    high_credit_score: { mean: 0.324166667, scale: 0.468062644 }
};

const coef = [
    0.0439554,   0.01014016, -0.12407531,  1.24631079,  0.05723121, -0.06052374,
    0.04368852,  2.54114295, -0.12182939, -0.55520801, -0.03030382, -0.04935772,
    0.02647809,  0.06341112, -0.05045196,  0.05568656, -0.04272741,  0.12545024,
   -0.11249109,  0.06690637,  0.33109613, -0.43513959,  0.05009625, -0.03534502,
    0.02062273,  0.02314419,  0.00453726, -0.07199436,  0.02508343, -0.08560619,
   -0.01968062,  0.1651569,   0.23271144, -0.46578081, -0.07613932,  0.32216785
];

const intercept = 0.03733884;

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.querySelector('button[type="submit"]');
    const resetBtn = document.querySelector('button[type="reset"]');
    
    const formContainer = document.getElementById('form-container');
    const resultContainer = document.getElementById('result-container');
    const btnBackToForm = document.getElementById('btn-back-to-form');
    const secureBadge = document.getElementById('secure-badge');

    if (!form) return;

    // 1. Intercept Submit Event and Run Predictions
    form.addEventListener('submit', function (event) {
        event.preventDefault();
        event.stopPropagation();

        if (!form.checkValidity() || !runCustomValidations()) {
            form.classList.add('was-validated');
            focusFirstInvalidInput();
        } else {
            // Form is valid: Show Loading Spinner and simulate calculation
            showLoadingSpinner();
            
            setTimeout(function() {
                // Gather data
                const data = getFormData();
                
                // Run inference
                const result = runInference(data);
                
                // Update UI elements with results
                updateResultsUI(data, result);
                
                // Transition to results screen
                transitionToResults();
            }, 1000); // 1-second visual feedback for professional feeling
        }
    }, false);

    // 2. Intercept Reset Trigger and Clear Validation Markers
    resetBtn.addEventListener('click', function () {
        form.classList.remove('was-validated');
        const customInvalidFields = form.querySelectorAll('.is-invalid');
        customInvalidFields.forEach(function (field) {
            field.classList.remove('is-invalid');
        });
    });

    // 3. Back to Form Transition
    btnBackToForm.addEventListener('click', function() {
        transitionToForm();
    });

    // 4. Custom Constraints & Validations
    function runCustomValidations() {
        let isValid = true;

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

    function markFieldInvalid(element, errorMessage) {
        element.classList.add('is-invalid');
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

    // 5. Gather Form Data
    function getFormData() {
        return {
            gender: document.getElementById('gender').value,
            age: parseInt(document.getElementById('age').value),
            family_status: document.getElementById('family_status').value,
            children_count: parseInt(document.getElementById('children_count').value) || 0,
            family_members: parseInt(document.getElementById('family_members').value) || 1,
            education: document.getElementById('education').value,
            owns_car: document.getElementById('owns_car').value,
            owns_property: document.getElementById('owns_property').value,
            annual_income: parseFloat(document.getElementById('annual_income').value),
            income_type: document.getElementById('income_type').value,
            years_employed: parseFloat(document.getElementById('years_employed').value),
            housing_type: document.getElementById('housing_type').value,
            credit_score: parseInt(document.getElementById('credit_score').value),
            existing_loans: parseInt(document.getElementById('existing_loans').value) || 0,
            debt_to_income: parseFloat(document.getElementById('debt_to_income').value) || 0.0,
            work_phone: document.getElementById('work_phone').checked ? 1 : 0,
            email: document.getElementById('email').checked ? 1 : 0
        };
    }

    // 6. Run Local Machine Learning Inference (Replicating Pipeline Preprocessing + Estimator Dot Product)
    function runInference(data) {
        let capped_annual_income = data.annual_income;
        let capped_years_employed = data.years_employed;
        let capped_debt_to_income = data.debt_to_income;

        // Custom Feature Engineering
        let income_per_family_member = capped_annual_income / (data.family_members === 0 ? 1 : data.family_members);
        let employment_to_age_ratio = capped_years_employed / (data.age === 0 ? 1 : data.age);
        let high_credit_score = data.credit_score >= 700 ? 1.0 : 0.0;

        // Numeric scaling
        const numeric = [
            data.children_count,
            capped_annual_income,
            data.age,
            capped_years_employed,
            data.work_phone,
            data.email,
            data.family_members,
            data.credit_score,
            data.existing_loans,
            capped_debt_to_income,
            income_per_family_member,
            employment_to_age_ratio,
            high_credit_score
        ];

        const num_keys = [
            'children_count',
            'annual_income',
            'age',
            'years_employed',
            'work_phone',
            'email',
            'family_members',
            'credit_score',
            'existing_loans',
            'debt_to_income',
            'income_per_family_member',
            'employment_to_age_ratio',
            'high_credit_score'
        ];

        let scaled_numeric = [];
        for (let i = 0; i < numeric.length; i++) {
            let key = num_keys[i];
            let mean = scaler_params[key].mean;
            let scale = scaler_params[key].scale;
            scaled_numeric.push((numeric[i] - mean) / scale);
        }

        // Categorical One-Hot Encoding (23 dimensions)
        let one_hot = new Array(23).fill(0.0);

        // Map gender
        if (data.gender === 'Female') one_hot[0] = 1.0;
        else if (data.gender === 'Male') one_hot[1] = 1.0;

        // Map owns_car
        if (data.owns_car === 'No') one_hot[2] = 1.0;
        else if (data.owns_car === 'Yes') one_hot[3] = 1.0;

        // Map owns_property
        if (data.owns_property === 'No') one_hot[4] = 1.0;
        else if (data.owns_property === 'Yes') one_hot[5] = 1.0;

        // Map income_type
        if (data.income_type === 'Commercial associate') one_hot[6] = 1.0;
        else if (data.income_type === 'Pensioner') one_hot[7] = 1.0;
        else if (data.income_type === 'State servant') one_hot[8] = 1.0;
        else if (data.income_type === 'Working') one_hot[9] = 1.0;

        // Map education
        if (data.education === 'Higher education') one_hot[10] = 1.0;
        else if (data.education === 'Incomplete higher') one_hot[11] = 1.0;
        else if (data.education === 'Lower secondary') one_hot[12] = 1.0;
        else if (data.education === 'Secondary') one_hot[13] = 1.0; // Maps to Secondary / Special Education

        // Map family_status
        if (data.family_status === 'Civil marriage') one_hot[14] = 1.0;
        else if (data.family_status === 'Married') one_hot[15] = 1.0;
        else if (data.family_status === 'Separated') one_hot[16] = 1.0;
        else if (data.family_status === 'Single') one_hot[17] = 1.0;
        else if (data.family_status === 'Widow') one_hot[18] = 1.0;

        // Map housing_type
        if (data.housing_type === 'House / apartment') one_hot[19] = 1.0;
        else if (data.housing_type === 'Municipal apartment') one_hot[20] = 1.0;
        else if (data.housing_type === 'Rented apartment') one_hot[21] = 1.0;
        else if (data.housing_type === 'With parents') one_hot[22] = 1.0;

        // Full preprocessed features array (36 inputs)
        let x = [...scaled_numeric, ...one_hot];

        // Linear dot product
        let z = intercept;
        for (let i = 0; i < x.length; i++) {
            z += x[i] * coef[i];
        }

        // Logistic sigmoid activation function
        let probability = 1.0 / (1.0 + Math.exp(-z));
        let decision = probability >= 0.5 ? "Approved" : "Rejected";

        return {
            probability: probability,
            decision: decision
        };
    }

    // 7. Update Results Cards and Log Fields
    function updateResultsUI(data, result) {
        const resultCardElement = document.getElementById('result-card-element');
        const resultBadge = document.getElementById('result-badge');
        const riskBadgeElement = document.getElementById('risk-badge-element');
        const decisionRecText = document.getElementById('decision-recommendation-text');
        const probTextElement = document.getElementById('probability-text-element');
        const progressBarElement = document.getElementById('progress-bar-element');

        // Text formatting
        const probPercentage = (result.probability * 100).toFixed(1) + "%";

        // Reset result card glow borders
        resultCardElement.classList.remove('glow-success', 'glow-danger');
        resultBadge.className = 'result-header-badge';
        riskBadgeElement.className = 'risk-badge';

        if (result.decision === "Approved") {
            resultCardElement.classList.add('glow-success');
            resultBadge.classList.add('bg-success');
            resultBadge.innerHTML = `<i class="fa-solid fa-circle-check me-2"></i>Approved`;
            
            decisionRecText.innerText = "Applicant meets all key creditworthiness parameters. Recommend approval for the requested credit facility. Set standard credit limit constraints.";
            
            progressBarElement.className = 'progress-bar prob-progress-bar bg-success';
        } else {
            resultCardElement.classList.add('glow-danger');
            resultBadge.classList.add('bg-danger');
            resultBadge.innerHTML = `<i class="fa-solid fa-circle-xmark me-2"></i>Rejected`;
            
            decisionRecText.innerText = "Applicant does not satisfy baseline approval constraints due to elevated risk projections (low credit score, high DTI ratio, or insufficient employment history). Recommend application rejection.";
            
            progressBarElement.className = 'progress-bar prob-progress-bar bg-danger';
        }

        // Set Risk Profile Badge
        if (result.probability >= 0.75) {
            riskBadgeElement.classList.add('bg-success', 'text-white');
            riskBadgeElement.innerText = "Low Risk";
        } else if (result.probability >= 0.50) {
            riskBadgeElement.classList.add('bg-warning', 'text-dark');
            riskBadgeElement.innerText = "Medium Risk";
        } else {
            riskBadgeElement.classList.add('bg-danger', 'text-white');
            riskBadgeElement.innerText = "High Risk";
        }

        // Progress indicators
        probTextElement.innerText = probPercentage;
        progressBarElement.style.width = '0%'; // Start at 0 for animation load
        progressBarElement.setAttribute('aria-valuenow', (result.probability * 100).toFixed(0));

        // Update Audit Table Fields
        document.getElementById('audit-gender').innerText = data.gender;
        document.getElementById('audit-age').innerText = data.age;
        document.getElementById('audit-marital-status').innerText = data.family_status;
        document.getElementById('audit-family-members').innerText = data.family_members;
        
        let eduLabel = data.education;
        if (data.education === 'Secondary') eduLabel = 'Secondary / High School';
        else if (data.education === 'Higher education') eduLabel = 'Higher Education / Degree';
        else if (data.education === 'Incomplete higher') eduLabel = 'Incomplete Higher';
        else if (data.education === 'Lower secondary') eduLabel = 'Lower Secondary';
        else if (data.education === 'Academic degree') eduLabel = 'Academic Degree';
        document.getElementById('audit-education').innerText = eduLabel;
        
        let housingLabel = data.housing_type;
        if (data.housing_type === 'House / apartment') housingLabel = 'House / Apartment';
        else if (data.housing_type === 'With parents') housingLabel = 'With Parents';
        else if (data.housing_type === 'Municipal apartment') housingLabel = 'Municipal Apartment';
        else if (data.housing_type === 'Rented apartment') housingLabel = 'Rented Apartment';
        else if (data.housing_type === 'Office apartment') housingLabel = 'Office Apartment';
        else if (data.housing_type === 'Co-op apartment') housingLabel = 'Co-op Apartment';
        document.getElementById('audit-housing').innerText = housingLabel;
        
        document.getElementById('audit-owns-car').innerText = data.owns_car;
        document.getElementById('audit-owns-property').innerText = data.owns_property;
        document.getElementById('audit-income').innerText = "$" + data.annual_income.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('audit-employment-length').innerText = data.years_employed;
        document.getElementById('audit-credit-score').innerText = data.credit_score;
        document.getElementById('audit-dti').innerText = (data.debt_to_income * 100).toFixed(1) + "%";
        document.getElementById('audit-active-loans').innerText = data.existing_loans;
        document.getElementById('audit-work-phone').innerText = data.work_phone === 1 ? "Yes" : "No";
        document.getElementById('audit-email').innerText = data.email === 1 ? "Yes" : "No";
        
        let incTypeLabel = data.income_type;
        if (data.income_type === 'Commercial associate') incTypeLabel = 'Commercial Associate';
        else if (data.income_type === 'State servant') incTypeLabel = 'State Servant';
        document.getElementById('audit-income-type').innerText = incTypeLabel;
    }

    // 8. Animation & Transitions Controllers
    function transitionToResults() {
        formContainer.classList.add('fade-out');
        
        setTimeout(function() {
            formContainer.classList.add('d-none');
            resultContainer.classList.remove('d-none');
            resultContainer.classList.add('fade-out');
            
            // Toggle Secure Badge to Audit Complete
            secureBadge.className = 'badge bg-dark border border-secondary text-secondary py-2 px-3';
            secureBadge.innerHTML = `<i class="fa-solid fa-lock me-2 text-info"></i>Audit Complete`;
            
            // Force redraw
            resultContainer.offsetHeight; 
            
            resultContainer.classList.remove('fade-out');
            
            // Animate progress bar fill-in
            setTimeout(function() {
                const progressBarElement = document.getElementById('progress-bar-element');
                const val = progressBarElement.getAttribute('aria-valuenow');
                progressBarElement.style.width = val + "%";
            }, 100);

            window.scrollTo({ top: 0, behavior: 'smooth' });
            restoreSubmitButton();
        }, 400);
    }

    function transitionToForm() {
        resultContainer.classList.add('fade-out');
        
        setTimeout(function() {
            resultContainer.classList.add('d-none');
            formContainer.classList.remove('d-none');
            formContainer.classList.add('fade-out');
            
            // Toggle Secure Badge back to Secure Portal
            secureBadge.className = 'badge bg-dark border border-secondary text-secondary py-2 px-3';
            secureBadge.innerHTML = `<i class="fa-solid fa-lock me-2 text-info"></i>Secure Portal`;
            
            // Force redraw
            formContainer.offsetHeight; 
            
            formContainer.classList.remove('fade-out');
            form.reset();
            form.classList.remove('was-validated');
            
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }, 400);
    }
});
