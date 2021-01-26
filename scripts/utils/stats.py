from stability_selection import RandomizedLogisticRegression


class PatchedRLG(RandomizedLogisticRegression):
    def __init__(
        self,
        weakness=0.5,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="liblinear",
        max_iter=100,
        multi_class="ovr",
        verbose=0,
        warm_start=False,
        n_jobs=1,
        callback_fn=None,
    ):
        super().__init__(
            weakness=weakness,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
        )
        self.callback_fn = callback_fn

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        if self.callback_fn is not None:
            self.callback_fn(self, X, y)
        return self
