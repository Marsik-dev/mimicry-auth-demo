import streamlit as st


def render(store) -> None:
    st.title("Mimicry Auth Demo")
    st.markdown("""
    **Биометрическая аутентификация по мимике лица (ГОСТ Р 52633.5-2011)**

    Система реализует нейросетевой преобразователь «биометрия–код» (НПБК):
    - видеопоток лица → признаки (геометрия, текстура, движение)
    - двухслойная нейронная сеть → стабильный криптографический код
    - аутентификация по расстоянию Хэмминга

    ---
    """)

    profiles = store.list_profiles()
    if profiles:
        st.success(f"Зарегистрировано профилей: {len(profiles)}")
        for uid in profiles:
            try:
                info = store.get_info(uid)
                st.write(f"• **{uid}** — стабильность: {info.quality_mean_stability:.2f}, "
                         f"FAR: {info.predicted_far:.2e}, "
                         f"собственных образцов: {info.n_own_samples}")
            except Exception:
                st.write(f"• {uid}")
    else:
        st.info("Нет зарегистрированных профилей. Перейдите в **Регистрация**.")

    st.markdown("---")
    st.markdown("**Навигация:** используй меню слева.")
